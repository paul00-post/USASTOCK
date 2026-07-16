"""
에이전트 A — 가치주 팩터 기반 Top 20 선별.

실행 주기:
  매일 8:00 : update_scores_agent_a()  — 3일 대기 관리 + 부분 재스코어링
  금요일 15:30 : friday_rerank_agent_a()  — 전체 재랭킹 + 스냅샷 저장
  반기 첫 월요일 : run_agent_a()  — KOSPI200 리밸런싱 + watchlist 20개 교체
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    BASE_DIR,
    SEMIANNUAL_MONTHS,
    VALUE_SECTORS,
)
from data.build_universe import get_universe_by_date
from data.dart_collector import load_dart_cache
from data.build_price_cache import load_price_cache
from data.fetch_shares import load_shares
from data.sector_classifier import load_sector_map
from data.build_factor_dataset import _get_kospi200_return_1m
from models.xgb_ranker import load_model, predict_scores
from utils.calendar_utils import (
    add_trading_days,
    count_trading_days,
)
from utils.logger import get_logger

logger = get_logger(__name__)

WATCHLIST_PATH    = BASE_DIR / "watchlist_cache.json"
PORTFOLIO_PATH    = BASE_DIR / "portfolio_state.json"
PENDING_PATH      = BASE_DIR / "data" / "pending_tickers_a.json"

# 섹터 종목 한도 (CLAUDE.md 섹터별 선정 한도 규칙)
_SECTOR_CAPS = {
    "철강금속": 2,
    "음식료":   2,
    "건설":     2,
    "운송창고": 2,
    "서비스":   1,
    "통신":     1,
    "전기가스": 1,
}


def _load_watchlist() -> dict:
    if WATCHLIST_PATH.exists():
        return json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
    return {"agent_a": [], "agent_b": [], "last_updated": None}


def _save_watchlist(wl: dict) -> None:
    WATCHLIST_PATH.write_text(json.dumps(wl, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_pending() -> dict:
    """pending_tickers_a.json 로드 — {ticker: wait_end_date} 형식."""
    if PENDING_PATH.exists():
        return json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    return {}


def _save_pending(pending: dict) -> None:
    PENDING_PATH.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")


def _select_top20(scored: pd.DataFrame, sector_map: pd.DataFrame) -> list[str]:
    """
    섹터 한도 초과 종목은 건너뛰고 다음 순위로 — 워터폴 방식.
    최대 20개. 20개 미만도 강제 채우지 않음.
    """
    sector_count: dict[str, int] = {}
    selected: list[str] = []

    for _, row in scored.sort_values("xgb_score", ascending=False).iterrows():
        if len(selected) >= 20:
            break
        ticker = row["ticker"]
        sec_row = sector_map[sector_map["ticker"] == ticker]
        sector = sec_row.iloc[0]["custom_sector"] if not sec_row.empty else None
        if sector not in VALUE_SECTORS:
            continue

        cap = _SECTOR_CAPS.get(sector)
        cnt = sector_count.get(sector, 0)
        if cap is not None and cnt >= cap:
            continue  # 섹터 한도 초과 — 다음 순위로

        selected.append(ticker)
        sector_count[sector] = cnt + 1

    return selected


def _score_tickers(
    tickers: list[str],
    ref_date: pd.Timestamp,
    sector_map: pd.DataFrame,
    shares_map: dict,
) -> pd.DataFrame:
    """지정 종목 목록 XGBoost 스코어링 → DataFrame(ticker, xgb_score) 반환."""
    from data.build_factor_dataset import (
        build_snapshot_row,
        _build_pools_for_date,
    )

    # 풀 배열 사전 계산
    sec_pools, all_pools, _, _ = _build_pools_for_date(
        tickers, ref_date, sector_map, shares_map, "A"
    )

    kospi_ret  = _get_kospi200_return_1m(ref_date)
    mktcap_map = {}
    for t in tickers:
        df_p = load_price_cache(t)
        sh   = shares_map.get(t)
        if df_p is not None and sh:
            df_p["Date"] = pd.to_datetime(df_p["Date"])
            row = df_p[df_p["Date"] < ref_date]
            if not row.empty:
                mktcap_map[t] = float(row.iloc[-1]["Close"]) * sh

    # 3개월 수익률 하위 10%
    ret_3m_list = []
    for t in tickers:
        df_p = load_price_cache(t)
        if df_p is None:
            continue
        df_p["Date"] = pd.to_datetime(df_p["Date"])
        p_now = df_p[df_p["Date"] < ref_date]
        p_3m  = df_p[df_p["Date"] < ref_date - pd.Timedelta(days=90)]
        if not p_now.empty and not p_3m.empty:
            ret_3m_list.append((p_now.iloc[-1]["Close"] - p_3m.iloc[-1]["Close"]) / p_3m.iloc[-1]["Close"])
    bottom10 = float(np.percentile(ret_3m_list, 10)) if ret_3m_list else -0.15

    rows = []
    for t in tickers:
        row = build_snapshot_row(
            t, ref_date, "A",
            sector_map, shares_map,
            sec_pools, all_pools, {}, {},
            kospi_ret, mktcap_map, bottom10,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker", "xgb_score"])

    df = pd.DataFrame(rows)
    model = load_model("A")
    if model is None:
        logger.warning("XGBoost A 모델 없음 — xgb_score=0.0")
        df["xgb_score"] = 0.0
        return df[["ticker", "xgb_score"]]

    from config.settings import AGENT_A_FEATURES
    missing = [c for c in AGENT_A_FEATURES if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    df["xgb_score"] = predict_scores(model, df, "A")
    return df[["ticker", "xgb_score"]]


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────

def update_scores_agent_a(ref_date: str | None = None) -> list[str]:
    """
    매일 8:00 실행.

    1. 새 공시 스캔 → pending 등록 (3일 대기)
    2. 기존 종목 days_since_earnings +1 (factor_dataset에서 관리)
    3. 3일 대기 완료 종목 부분 재스코어링 → Top 20 미세 조정
    """
    if ref_date is None:
        ref_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    ref = pd.Timestamp(ref_date)

    sector_map  = load_sector_map()
    shares_map  = load_shares()
    wl          = _load_watchlist()
    pending     = _load_pending()

    # 현재 유니버스
    try:
        universe = get_universe_by_date(ref_date)
    except RuntimeError as e:
        logger.error("유니버스 로드 실패: %s", e)
        return wl.get("agent_a", [])

    # 새 공시 스캔 — publish_date가 최근 3 영업일 이내인 종목
    newly_announced: list[str] = []
    for ticker in universe:
        dart_df = load_dart_cache(ticker)
        if dart_df is None or dart_df.empty:
            continue
        dart_df["publish_date"] = pd.to_datetime(dart_df["publish_date"], errors="coerce")
        valid = dart_df[dart_df["publish_date"].notna() & (dart_df["publish_date"] <= ref)]
        if valid.empty:
            continue
        pub_date = valid.sort_values("publish_date").iloc[-1]["publish_date"]
        days_since = count_trading_days(pub_date, ref)
        if days_since <= 3 and ticker not in pending:
            # 3일 대기 시작
            wait_end = add_trading_days(pub_date, 2)
            pending[ticker] = wait_end.strftime("%Y-%m-%d")
            newly_announced.append(ticker)

    if newly_announced:
        logger.info("신규 공시 %d종목 → 3일 대기 등록: %s", len(newly_announced), newly_announced[:5])

    # 3일 대기 완료 종목 파악
    completed = [
        t for t, wait_end in list(pending.items())
        if ref >= pd.Timestamp(wait_end)
    ]
    for t in completed:
        del pending[t]

    _save_pending(pending)

    if completed:
        logger.info("3일 대기 완료 %d종목 재스코어링: %s", len(completed), completed[:5])
        # 완료 종목만 재스코어링
        scored = _score_tickers(completed, ref, sector_map, shares_map)
        # 기존 watchlist와 병합하여 Top 20 갱신
        existing_scores = _score_tickers(
            [t for t in wl.get("agent_a", []) if t not in completed],
            ref, sector_map, shares_map,
        )
        all_scored = pd.concat([existing_scores, scored], ignore_index=True)
        new_top20  = _select_top20(all_scored, sector_map)
        wl["agent_a"] = new_top20
        wl["last_updated"] = ref_date
        _save_watchlist(wl)
        logger.info("Agent A Top 20 미세 조정 완료: %s", new_top20[:5])

    return wl.get("agent_a", [])


def friday_rerank_agent_a(ref_date: str | None = None) -> list[str]:
    """
    금요일 15:30 실행.
    금요일 종가 기준 전체 재랭킹 → Top 20 재확정 + 스냅샷 저장.
    """
    if ref_date is None:
        ref_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    ref = pd.Timestamp(ref_date)

    sector_map = load_sector_map()
    shares_map = load_shares()
    wl         = _load_watchlist()

    try:
        universe = get_universe_by_date(ref_date)
    except RuntimeError as e:
        logger.error("유니버스 로드 실패: %s", e)
        return wl.get("agent_a", [])

    # 전 종목 재스코어링
    logger.info("Agent A 전체 재랭킹 시작 (%s, %d 종목)", ref_date, len(universe))
    scored = _score_tickers(universe, ref, sector_map, shares_map)
    top20  = _select_top20(scored, sector_map)

    wl["agent_a"]    = top20
    wl["last_updated"] = ref_date
    _save_watchlist(wl)

    logger.info("Agent A Top 20 재확정: %s", top20)
    return top20


def run_agent_a(ref_date: str | None = None) -> list[str]:
    """
    반기 첫 월요일 (1월·7월 첫째 월요일) 실행.
    KOSPI200 유니버스 리밸런싱 + watchlist 20개 교체.
    보유 중인 종목은 강제 청산 없이 자연 청산 대기.
    """
    if ref_date is None:
        ref_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    logger.info("Agent A 반기 리밸런싱 시작: %s", ref_date)
    top20 = friday_rerank_agent_a(ref_date)

    # 보유 종목 예외처리 로그
    if PORTFOLIO_PATH.exists():
        portfolio = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
        held = list(portfolio.get("holdings", {}).keys())
        excluded = [t for t in held if t not in top20]
        if excluded:
            logger.info(
                "새 watchlist 미포함 보유 종목 (자연 청산 대기): %s", excluded
            )

    return top20
