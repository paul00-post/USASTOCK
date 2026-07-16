"""
지수 구성종목 리스트 관리 — 코스피200 / S&P500 / 나스닥100.

S&P500·나스닥100은 깔끔한 무료 API가 없어 위키피디아 표를 긁어오는 방식을 쓴다.
구성종목은 자주 안 바뀌므로(연 몇 차례) 주간 캐시로 충분하다.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from config.settings import DATA_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = DATA_DIR / "index_constituents"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_REFRESH_DAYS = 7  # 주 1회 갱신

_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NDX100_WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _load_cache(name: str) -> dict | None:
    path = _cache_path(name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        updated = datetime.fromisoformat(data["updated_at"])
        if datetime.now() - updated < timedelta(days=_REFRESH_DAYS):
            return data
    except Exception:
        pass
    return None


def _read_wiki_tables(url: str) -> list[pd.DataFrame]:
    """위키피디아는 User-Agent 없는 요청을 차단하므로 헤더를 붙여 직접 받아온다."""
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (quant-data-collector)"}, timeout=15)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))


def _save_cache(name: str, tickers: list[str]) -> None:
    _cache_path(name).write_text(
        json.dumps({"updated_at": datetime.now().isoformat(), "tickers": tickers}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_sp500_tickers(force_refresh: bool = False) -> list[str]:
    """S&P500 구성종목 티커 목록 (위키피디아 표, 주간 캐시)."""
    if not force_refresh:
        cached = _load_cache("sp500")
        if cached:
            return cached["tickers"]

    try:
        tables = _read_wiki_tables(_SP500_WIKI_URL)
        df = tables[0]
        tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        _save_cache("sp500", tickers)
        logger.info("S&P500 구성종목 갱신: %d종목", len(tickers))
        return tickers
    except Exception as e:
        logger.error("S&P500 구성종목 조회 실패: %s", e)
        cached = _load_cache("sp500")
        if cached:
            logger.warning("이전 캐시로 폴백 (마지막 갱신: %s)", cached["updated_at"])
            return cached["tickers"]
        return []


def get_nasdaq100_tickers(force_refresh: bool = False) -> list[str]:
    """나스닥100 구성종목 티커 목록 (위키피디아 표, 주간 캐시)."""
    if not force_refresh:
        cached = _load_cache("nasdaq100")
        if cached:
            return cached["tickers"]

    try:
        tables = _read_wiki_tables(_NDX100_WIKI_URL)
        # 위키 문서 내 표 순서가 바뀔 수 있어 "Ticker" 컬럼이 있는 표를 찾는다.
        target = None
        for t in tables:
            cols = [str(c) for c in t.columns]
            if any("ticker" in c.lower() for c in cols):
                target = t
                break
        if target is None:
            raise ValueError("나스닥100 구성종목 표를 찾지 못함")
        ticker_col = [c for c in target.columns if "ticker" in str(c).lower()][0]
        tickers = target[ticker_col].astype(str).str.replace(".", "-", regex=False).tolist()
        _save_cache("nasdaq100", tickers)
        logger.info("나스닥100 구성종목 갱신: %d종목", len(tickers))
        return tickers
    except Exception as e:
        logger.error("나스닥100 구성종목 조회 실패: %s", e)
        cached = _load_cache("nasdaq100")
        if cached:
            logger.warning("이전 캐시로 폴백 (마지막 갱신: %s)", cached["updated_at"])
            return cached["tickers"]
        return []


def get_kospi200_tickers(ref_date: str | pd.Timestamp | None = None) -> list[str]:
    """코스피200 구성종목 — 기존 유니버스 스냅샷 재사용."""
    from data.build_universe import get_universe_by_date

    if ref_date is None:
        ref_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    try:
        return get_universe_by_date(ref_date)
    except RuntimeError as e:
        logger.error("코스피200 유니버스 로드 실패: %s", e)
        return []
