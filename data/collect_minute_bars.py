"""
분봉 데이터 일별 수집기 — 코스피200 / S&P500 / 나스닥100 / 3개 지수(ETF 대용).

다른 프로그램에서 재사용할 목적으로 이 프로젝트와 독립적으로 저장한다.
장 마감 후(국내 18:00 이후, 미국 06:00 KST 이후)부터 다음 개장 전까지
아무 때나 실행하면 된다 — 장중이 아니므로 API 호출 제한에 여유롭다.

저장 위치: data/minute_bars/{market}/{YYYYMMDD}/{ticker}.parquet
  market ∈ {kospi200, sp500, nasdaq100, index_kospi200, index_sp500, index_nasdaq100}

실행: python -m data.collect_minute_bars [YYYY-MM-DD]
      인자 없으면 오늘 날짜로 수집
"""

from __future__ import annotations

import sys
import time as time_module
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR
from data.index_constituents import (
    get_kospi200_tickers,
    get_nasdaq100_tickers,
    get_sp500_tickers,
)
from utils.kis_client import _base_url, get_access_token
from utils.logger import get_logger

import requests
from config.settings import KIS_CONFIG

logger = get_logger(__name__)

MINUTE_BAR_DIR = DATA_DIR / "minute_bars"

# 초당 호출 제한 대비 각 페이지 호출 사이 최소 간격 (초)
_CALL_DELAY = 0.25
_MAX_RETRIES = 3

# 지수 자체는 대표 ETF로 대체 조회 (지수는 거래 상품이 아니라 API 대상이 아닐 수 있음)
INDEX_PROXIES = {
    "index_kospi200":  ("069500", "domestic"),   # KODEX 200
    "index_sp500":     ("SPY",    "NYS"),
    "index_nasdaq100": ("QQQ",    "NAS"),
}


def _domestic_page(ticker: str, hour: str) -> list[dict] | None:
    token = get_access_token()
    if token is None:
        return None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(
                f"{_base_url()}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
                headers={
                    "content-type": "application/json; charset=utf-8",
                    "authorization": f"Bearer {token}",
                    "appkey": KIS_CONFIG["app_key"], "appsecret": KIS_CONFIG["app_secret"],
                    "tr_id": "FHKST03010200", "custtype": "P",
                },
                params={
                    "FID_ETC_CLS_CODE": "", "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": ticker, "FID_INPUT_HOUR_1": hour,
                    "FID_PW_DATA_INCU_YN": "Y",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") != "0":
                raise RuntimeError(data.get("msg1"))
            return data.get("output2", [])
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                time_module.sleep(_CALL_DELAY * (attempt + 1))
                continue
            logger.warning("%s 국내 분봉 페이지 실패 (hour=%s): %s", ticker, hour, e)
            return None


def fetch_domestic_minute_bars(ticker: str, max_pages: int = 20) -> pd.DataFrame | None:
    """
    국내 종목 당일 1분봉 전체 수집 (09:00~15:30, 페이지당 30건).
    hour='' 로 시작해 직전 배치의 마지막 시각으로 계속 이어받는다.
    """
    all_rows: list[dict] = []
    hour = ""
    seen_hours: set[str] = set()

    for _ in range(max_pages):
        rows = _domestic_page(ticker, hour)
        if not rows:
            break
        new_rows = [r for r in rows if r["stck_cntg_hour"] not in seen_hours]
        if not new_rows:
            break
        all_rows.extend(new_rows)
        seen_hours.update(r["stck_cntg_hour"] for r in new_rows)

        earliest = rows[-1]["stck_cntg_hour"]
        if earliest <= "090000":  # 장 시작 시각까지 도달
            break
        hour = earliest
        time_module.sleep(_CALL_DELAY)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows).rename(columns={
        "stck_bsop_date": "date", "stck_cntg_hour": "time",
        "stck_oprc": "open", "stck_hgpr": "high",
        "stck_lwpr": "low", "stck_prpr": "close", "cntg_vol": "volume",
    })
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df.sort_values("time").reset_index(drop=True)


def _overseas_page(ticker: str, excd: str, keyb: str) -> tuple[list[dict], str] | None:
    token = get_access_token()
    if token is None:
        return None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(
                f"{_base_url()}/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice",
                headers={
                    "content-type": "application/json; charset=utf-8",
                    "authorization": f"Bearer {token}",
                    "appkey": KIS_CONFIG["app_key"], "appsecret": KIS_CONFIG["app_secret"],
                    "tr_id": "HHDFS76950200", "custtype": "P",
                },
                params={
                    "AUTH": "", "EXCD": excd, "SYMB": ticker,
                    "NMIN": "1", "PINC": "1", "NEXT": "1" if keyb else "",
                    "NREC": "120", "FILL": "", "KEYB": keyb,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") != "0":
                raise RuntimeError(data.get("msg1"))
            rows = data.get("output2", [])
            next_keyb = data.get("output1", {}).get("ektm", "")
            return rows, next_keyb
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                time_module.sleep(_CALL_DELAY * (attempt + 1))
                continue
            logger.warning("%s(%s) 해외 분봉 페이지 실패: %s", ticker, excd, e)
            return None


def fetch_overseas_minute_bars(ticker: str, excd: str, max_pages: int = 6) -> pd.DataFrame | None:
    """
    해외 종목 당일 1분봉 전체 수집. 페이지당 최대 120건(NREC)까지 요청 가능.
    output1.ektm(장 시작 한국시각)을 KEYB로 넘겨 이어받는다.
    """
    all_rows: list[dict] = []
    keyb = ""

    for _ in range(max_pages):
        result = _overseas_page(ticker, excd, keyb)
        if result is None:
            break
        rows, next_keyb = result
        if not rows:
            break
        all_rows.extend(rows)
        if not next_keyb or next_keyb == keyb:
            break
        keyb = next_keyb
        time_module.sleep(_CALL_DELAY)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows).rename(columns={
        "tymd": "us_date", "xhms": "us_time", "kymd": "kr_date", "khms": "kr_time",
        "open": "open", "high": "high", "low": "low", "last": "close", "evol": "volume",
    })
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df.drop_duplicates(subset=["us_date", "us_time"]).sort_values("us_time").reset_index(drop=True)


def _save(market: str, date_str: str, ticker: str, df: pd.DataFrame) -> None:
    out_dir = MINUTE_BAR_DIR / market / date_str.replace("-", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{ticker}.parquet", index=False)


def collect_market(market: str, tickers: list[str], is_overseas: bool, excd: str = "", date_str: str | None = None) -> int:
    """지정 시장의 전 종목 당일 분봉을 수집·저장. 저장된 종목 수 반환."""
    if date_str is None:
        date_str = pd.Timestamp.today().strftime("%Y-%m-%d")

    n_saved = 0
    for i, ticker in enumerate(tickers, 1):
        df = (
            fetch_overseas_minute_bars(ticker, excd) if is_overseas
            else fetch_domestic_minute_bars(ticker)
        )
        if df is None or df.empty:
            logger.debug("[%d/%d] %s(%s) 분봉 없음 — 스킵", i, len(tickers), market, ticker)
            continue
        _save(market, date_str, ticker, df)
        n_saved += 1
        if i % 20 == 0:
            logger.info("[%s] %d/%d종목 진행 중...", market, i, len(tickers))

    logger.info("[%s] 수집 완료: %d/%d종목 저장 (%s)", market, n_saved, len(tickers), date_str)
    return n_saved


def run_daily_minute_bar_collection(date_str: str | None = None, include_overseas: bool = False) -> dict:
    """
    분봉 수집 — 장 마감 후 아무 때나 실행 가능.

    기본값은 코스피200 + 코스피200 지수(KODEX200)만 수집한다 (완전히 검증됨).
    S&P500/나스닥100은 해외 분봉 API의 과거 조회 깊이가 아직 불확실해서
    (한 번에 20분 정도만 나오고 더 과거로 못 감) include_overseas=True로
    명시했을 때만 시도한다 — 미국 장중(22:30~05:00 KST)에 재검증 필요.
    """
    if date_str is None:
        date_str = pd.Timestamp.today().strftime("%Y-%m-%d")

    logger.info("=== 분봉 데이터 수집 시작: %s (해외 포함=%s) ===", date_str, include_overseas)
    results: dict[str, int] = {}

    results["kospi200"] = collect_market(
        "kospi200", get_kospi200_tickers(date_str), is_overseas=False, date_str=date_str,
    )
    n = collect_market("index_kospi200", [INDEX_PROXIES["index_kospi200"][0]], is_overseas=False, date_str=date_str)
    results["index_kospi200"] = n

    if include_overseas:
        results["sp500"] = collect_market(
            "sp500", get_sp500_tickers(), is_overseas=True, excd="NYS", date_str=date_str,
        )
        results["nasdaq100"] = collect_market(
            "nasdaq100", get_nasdaq100_tickers(), is_overseas=True, excd="NAS", date_str=date_str,
        )
        for market in ("index_sp500", "index_nasdaq100"):
            ticker, excd = INDEX_PROXIES[market]
            results[market] = collect_market(market, [ticker], is_overseas=True, excd=excd, date_str=date_str)

    logger.info("=== 분봉 데이터 수집 완료: %s | %s ===", date_str, results)
    return results


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--overseas"]
    date_arg = args[0] if args else None
    run_daily_minute_bar_collection(date_arg, include_overseas="--overseas" in sys.argv)
