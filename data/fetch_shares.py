"""
발행주식수 수집기 (파일명은 국내 프로젝트 호환용으로 유지 — agents/agent_b.py,
data/build_factor_dataset.py가 load_shares를 그대로 import).

소스: SEC XBRL companyfacts의 dei:EntityCommonStockSharesOutstanding —
dart_collector.py가 이미 받아둔 companyfacts 원본 캐시(data/sec_raw_cache/)를
그대로 재사용한다(같은 회사 데이터를 두 번 받을 필요 없음). 가장 최근 보고된
값을 "현재 발행주식수"로 사용.

실행: python -m data.fetch_shares
      (인자는 무시됨 — 현재 발행주식수만 제공, 국내 버전과 동일)
"""

from __future__ import annotations

import json

from config.settings import DATA_DIR
from data.dart_collector import fetch_companyfacts, get_cik_with_fallback
from utils.logger import get_logger

logger = get_logger(__name__)

SHARES_PATH = DATA_DIR / "sec_shares.json"


def _latest_shares_outstanding(companyfacts: dict) -> int | None:
    """dei:EntityCommonStockSharesOutstanding 중 가장 최근(filed 최댓값) 값."""
    dei = companyfacts.get("facts", {}).get("dei", {})
    facts = dei.get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    if not facts:
        return None
    latest = max(facts, key=lambda f: f.get("filed", ""))
    return int(latest["val"])


def fetch_shares_one(ticker: str) -> int | None:
    """단일 종목 발행주식수 조회."""
    cik = get_cik_with_fallback(ticker)
    if cik is None:
        return None
    companyfacts = fetch_companyfacts(cik, ticker)
    if companyfacts is None:
        return None
    return _latest_shares_outstanding(companyfacts)


def fetch_shares(tickers: list[str] | None = None) -> dict[str, int]:
    """
    발행주식수 수집. tickers=None이면 S&P500 역사적 합집합 전체.
    """
    if tickers is None:
        from data.build_universe import get_all_tickers_until
        import datetime
        tickers = get_all_tickers_until(datetime.date.today().year)
        logger.info("대상: S&P500 역사적 합집합 %d 종목", len(tickers))

    shares_map: dict[str, int] = {}
    for i, ticker in enumerate(tickers, 1):
        shares = fetch_shares_one(ticker)
        if shares:
            shares_map[ticker] = shares
        if i % 50 == 0:
            logger.info("발행주식수 조회 진행: %d/%d", i, len(tickers))

    logger.info("발행주식수 %d/%d 종목 수집 완료", len(shares_map), len(tickers))
    return shares_map


def build_shares_json(ref_date: str | None = None) -> None:
    """발행주식수 JSON 저장. ref_date는 국내 버전과의 호환용 인자로 무시됨."""
    shares = fetch_shares()
    SHARES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHARES_PATH.write_text(
        json.dumps(shares, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("발행주식수 저장: %s (%d 종목)", SHARES_PATH, len(shares))


def load_shares() -> dict[str, int]:
    """저장된 발행주식수 로드."""
    if not SHARES_PATH.exists():
        raise FileNotFoundError(
            "sec_shares.json 없음. python -m data.fetch_shares 를 먼저 실행하세요."
        )
    return json.loads(SHARES_PATH.read_text(encoding="utf-8"))


def get_shares(ticker: str) -> int | None:
    """단일 종목 발행주식수 반환."""
    try:
        return load_shares().get(ticker)
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    build_shares_json()
