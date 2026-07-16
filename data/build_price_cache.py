"""
일별 OHLCV 가격 캐시 빌더 — yfinance 우선, 상장폐지·개명 종목은 Tiingo로 보충.

배경(2026-07-16 결정 과정):
  yfinance는 무료·무제한급이지만 인수합병·상장폐지된 종목의 과거 가격 데이터를
  거의 안 갖고 있다(실측 17개 표본 중 12%만 성공). 반면 Tiingo 무료 티어는 같은
  표본에서 67%가 성공했고, 종목 메타데이터에 "DELISTED" 여부와 정확한 마지막
  거래일(endDate)까지 준다 — 이게 실제 인수합병 완료일과 정확히 일치하는 것도
  확인됐다(Celgene→BMS 2019-11-22, Allergan→AbbVie 2020-05-08).

  Tiingo는 월 500종목 제한이 있지만, "yfinance가 이미 되는 대부분의 현재
  활동 종목"에는 아예 안 쓰고 "yfinance가 실패한 종목"에만 쓰므로 전체
  유니버스(500+종목)를 다 Tiingo로 받는 것보다 훨씬 적은 호출만 필요하다.

소스 우선순위: ① yfinance(무료, 무제한급) → ② Tiingo(무료 티어, 실패시만)

출력:
  data/price_cache/{ticker}.parquet       — Date, Open, High, Low, Close, Volume
  data/price_cache_meta/{ticker}.json     — {source, start_date, end_date, is_delisted}
    is_delisted=True인 종목은 end_date 이후로는 원천적으로 데이터가 없다는 뜻
    (조회 실패와 구분하기 위한 메타데이터 — 포지션 관리 로직에서 참고).

실행: python -m data.build_price_cache [ticker ...]
      인자 없으면 S&P500 역사적 합집합 전체 수집 (data.build_universe 기준)
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd
import requests
import yfinance as yf

from config.settings import DATA_DIR, PRICE_START_DATE, TIINGO_API_KEY
from utils.logger import get_logger

logger = get_logger(__name__)

PRICE_CACHE_DIR = DATA_DIR / "price_cache"
META_CACHE_DIR  = DATA_DIR / "price_cache_meta"

# Tiingo 무료 티어 실제 제한은 시간당 50회(=최소 72초 간격) — 2026-07-16
# 처음엔 2초로 잘못 넣어놨었음(그러면 이론상 시간당 1800회라 몇 분 만에
# 차단당함). 75초로 여유를 둬서 정확히 지킨다.
_TIINGO_DELAY = 75.0

# 종목당 메타(startDate/endDate) API를 따로 부르면 호출이 2배로 늘어나
# 시간당 처리량이 반토막난다 — 대신 실제 받아온 가격 데이터의 마지막 날짜가
# "오늘"보다 이만큼(일) 오래됐으면 상장폐지로 간주한다(별도 API 호출 없이
# 같은 정보를 근사). 정확한 마지막 거래일 자체는 어차피 가격 데이터의
# 마지막 행이 그대로 알려준다.
_STALE_DAYS_THRESHOLD = 14


# ── ① yfinance ────────────────────────────────────────────────────────────────

def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as e:
        logger.debug("%s yfinance 예외: %s", ticker, e)
        return None
    if df is None or df.empty:
        return None
    # yfinance 최신 버전은 단일 종목이어도 MultiIndex 컬럼(('Close','AAPL') 등)을 반환
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ── ② Tiingo (yfinance 실패 시 폴백) ──────────────────────────────────────────

def _tiingo_headers() -> dict:
    if not TIINGO_API_KEY:
        raise RuntimeError("TIINGO_API_KEY 미설정 — .env에 채우세요 (무료 가입: https://www.tiingo.com).")
    return {"Content-Type": "application/json"}


def _fetch_tiingo_prices(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        time.sleep(_TIINGO_DELAY)
        r = requests.get(
            f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
            params={"startDate": start, "endDate": end, "token": TIINGO_API_KEY},
            headers=_tiingo_headers(), timeout=30,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        df = pd.DataFrame(data)
        # adjOpen/adjHigh/adjLow/adjClose/adjVolume = 배당·액면분할 반영 수정주가
        # (미국 대형주는 분할이 잦아 원본가(open/close 등) 대신 반드시 조정가 사용)
        df = df.rename(columns={
            "date": "Date", "adjOpen": "Open", "adjHigh": "High",
            "adjLow": "Low", "adjClose": "Close", "adjVolume": "Volume",
        })[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df
    except Exception as e:
        logger.debug("%s Tiingo 가격 조회 실패: %s", ticker, e)
        return None


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────

def fetch_ohlcv(
    ticker: str, start: str = PRICE_START_DATE, end: str | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    """
    ticker의 OHLCV를 yfinance 우선, 실패 시 Tiingo로 수집.

    Returns
    -------
    (DataFrame 또는 None, meta dict) — meta: {"source": "yfinance"|"tiingo"|None,
    "is_delisted": bool, "tiingo_end_date": str|None}
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    df = _fetch_yfinance(ticker, start, end)
    if df is not None and not df.empty:
        return df, {"source": "yfinance", "is_delisted": False, "tiingo_end_date": None}

    logger.debug("%s yfinance 실패 — Tiingo 폴백 시도", ticker)
    df = _fetch_tiingo_prices(ticker, start, end)
    if df is not None and not df.empty:
        last_date = df["Date"].max()
        days_stale = (pd.Timestamp.today().normalize() - last_date).days
        is_delisted = days_stale > _STALE_DAYS_THRESHOLD
        return df, {
            "source": "tiingo",
            "is_delisted": is_delisted,
            "tiingo_end_date": last_date.strftime("%Y-%m-%d") if is_delisted else None,
        }

    logger.warning("%s: yfinance·Tiingo 둘 다 실패", ticker)
    return None, {"source": None, "is_delisted": False, "tiingo_end_date": None}


def build_price_cache(
    tickers: list[str] | None = None,
    start: str = PRICE_START_DATE,
    update_only: bool = True,
) -> None:
    """
    종목별 OHLCV parquet + 메타 JSON 저장.

    tickers=None이면 S&P500 역사적 합집합(생존편향 방지) 전체 수집.
    update_only=True면 기존 캐시 마지막 날짜 이후만 갱신 — 단, 이미 상장폐지로
    확인된 종목(메타에 is_delisted=True)은 더 조회할 미래 데이터가 없으므로 스킵.
    """
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    META_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        from data.build_universe import get_all_tickers_until
        tickers = get_all_tickers_until(pd.Timestamp.today().year)
        logger.info("S&P500 역사적 합집합 전체 수집: %d 종목", len(tickers))

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    failed: list[str] = []

    for i, ticker in enumerate(tickers, 1):
        path = PRICE_CACHE_DIR / f"{ticker}.parquet"
        meta_path = META_CACHE_DIR / f"{ticker}.json"
        fetch_start = start

        existing_meta: dict = {}
        if meta_path.exists():
            try:
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                existing_meta = {}

        if update_only and path.exists():
            if existing_meta.get("is_delisted"):
                logger.debug("[%d/%d] %s 상장폐지 확인됨 — 갱신 스킵", i, len(tickers), ticker)
                continue
            try:
                existing = pd.read_parquet(path)
                last_date = pd.to_datetime(existing["Date"]).max()
                fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if fetch_start > today:
                    logger.debug("[%d/%d] %s 최신 상태 — 스킵", i, len(tickers), ticker)
                    continue
            except Exception:
                fetch_start = start

        logger.info("[%d/%d] %s 가격 수집: %s~%s", i, len(tickers), ticker, fetch_start, today)
        df_new, meta = fetch_ohlcv(ticker, start=fetch_start, end=today)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        if df_new is None:
            failed.append(ticker)
            continue

        if update_only and path.exists():
            df_old = pd.read_parquet(path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
        else:
            df_combined = df_new

        df_combined.to_parquet(path, index=False)
        if meta.get("is_delisted"):
            logger.info("%s: 상장폐지 확인(source=%s, 마지막 거래일=%s)", ticker, meta["source"], meta.get("tiingo_end_date"))

    if failed:
        logger.warning("가격 수집 실패 종목 (%d개): %s", len(failed), failed)
    logger.info("가격 캐시 빌드 완료")


def load_price_cache(ticker: str) -> pd.DataFrame | None:
    """캐시된 OHLCV parquet 로드."""
    path = PRICE_CACHE_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_price_meta(ticker: str) -> dict | None:
    """캐시된 메타(소스·상장폐지 여부) 로드."""
    path = META_CACHE_DIR / f"{ticker}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_prices_before(ticker: str, ref_date: str | pd.Timestamp, n: int) -> pd.DataFrame | None:
    """
    ref_date 직전 영업일 기준 최근 n 행 반환.
    룩어헤드 방지: ref_date 당일 미포함 (< 기준).
    """
    df = load_price_cache(ticker)
    if df is None:
        return None
    ref = pd.Timestamp(ref_date)
    past = df[df["Date"] < ref].tail(n)
    return past if not past.empty else None


if __name__ == "__main__":
    args = sys.argv[1:]
    build_price_cache(tickers=args if args else None)
