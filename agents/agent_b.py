"""
에이전트 B — 성장주 팩터 기반 Top 30 선별 + factor_dataset_B.parquet 갱신.

실행 주기:
  매일 8:00 : update_scores_agent_b()  — 3일 대기 관리 + 부분 재스코어링
  금요일 15:30 : friday_rerank_agent_b()  — 전체 재랭킹 + 스냅샷 저장

에이전트 A는 더 이상 사용하지 않으므로, 유니버스는 "가치/성장 섹터 분리" 없이
코스피200 전체 종목을 대상으로 한다 (섹터 필터 제거, Top 20 → Top 30).

[핵심 수정 사항]
이전 버전은 존재하지 않는 함수(build_snapshot_row, _build_pools_for_date)를
불러서 매번 크래시가 났다. 실제로 존재하는 배치 계산 로직
(data.build_factor_dataset._build_pools/_build_row, 백테스팅에도 쓰이는
검증된 코드)을 재사용하는 build_live_snapshot()으로 교체했다.
그리고 계산된 피처를 factor_dataset_B.parquet에 실제로 저장하도록 했다 —
이전에는 점수만 계산하고 버렸기 때문에, 에이전트 C가 읽는 그 파일은
한 번도 갱신되지 않고 고정된 채로 남아있었다.

3일 대기 로직(공시 후 3영업일 지나야 ear_3d 등 피처가 완성됨)은 그대로 유지한다.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config.settings import BASE_DIR
from data.build_universe import get_universe_by_date
from data.dart_collector import load_dart_cache
from data.build_factor_dataset import _save_snapshot_append, build_live_snapshot
from data.fetch_shares import load_shares
from data.sector_classifier import load_sector_map
from models.xgb_ranker import load_model, predict_scores
from utils.calendar_utils import add_trading_days, count_trading_days
from utils.logger import get_logger

logger = get_logger(__name__)

WATCHLIST_PATH = BASE_DIR / "watchlist_cache.json"
PORTFOLIO_PATH = BASE_DIR / "portfolio_state.json"
PENDING_PATH   = BASE_DIR / "data" / "pending_tickers_b.json"

POOL_SIZE = 30  # Agent C의 POOL_N과 동일하게 맞춤 (기존 20 → 30)


def _load_watchlist() -> dict:
    if WATCHLIST_PATH.exists():
        return json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
    return {"agent_b": [], "last_updated": None}


def _save_watchlist(wl: dict) -> None:
    WATCHLIST_PATH.write_text(json.dumps(wl, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_pending() -> dict:
    if PENDING_PATH.exists():
        return json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    return {}


def _save_pending(pending: dict) -> None:
    PENDING_PATH.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")


def _select_top_n(scored: pd.DataFrame, n: int = POOL_SIZE) -> list[str]:
    """xgb_score 상위 n개 선택. 에이전트 A 폐기로 섹터 제한 없이 코스피200 전체 대상."""
    if scored.empty:
        return []
    return scored.sort_values("xgb_score", ascending=False).head(n)["ticker"].tolist()


def _score_tickers_b(tickers: list[str], ref_date: pd.Timestamp) -> pd.DataFrame:
    """
    지정 종목 피처를 실제로 계산(build_live_snapshot)하고 factor_dataset_B.parquet에
    저장한 뒤, 학습된 모델로 점수를 매겨 (ticker, xgb_score) 반환한다.
    """
    snapshot = build_live_snapshot("B", ref_date, tickers)
    if snapshot.empty:
        return pd.DataFrame(columns=["ticker", "xgb_score"])

    _save_snapshot_append("B", snapshot)

    model = load_model("B")
    if model is None:
        logger.warning("XGBoost B 모델 없음 — xgb_score=0.0")
        snapshot = snapshot.copy()
        snapshot["xgb_score"] = 0.0
        return snapshot[["ticker", "xgb_score"]]

    from config.settings import AGENT_B_FEATURES
    df = snapshot.copy()
    missing = [c for c in AGENT_B_FEATURES if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    df["xgb_score"] = predict_scores(model, df, "B")
    return df[["ticker", "xgb_score"]]


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────

def update_scores_agent_b(ref_date: str | None = None) -> list[str]:
    """
    매일 8:00 실행. 3일 대기 관리 + 부분 재스코어링.
    3일 대기가 완료된 종목만 factor_dataset_B.parquet에 새 값으로 갱신된다
    (완료 안 된 종목은 이전에 저장된 값이 그대로 남아있음 — 정상 동작).
    """
    if ref_date is None:
        ref_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    ref = pd.Timestamp(ref_date)

    wl      = _load_watchlist()
    pending = _load_pending()

    try:
        universe = get_universe_by_date(ref_date)
    except RuntimeError as e:
        logger.error("유니버스 로드 실패: %s", e)
        return wl.get("agent_b", [])

    newly_announced: list[str] = []
    for ticker in universe:
        dart_df = load_dart_cache(ticker)
        if dart_df is None or dart_df.empty:
            continue
        dart_df["publish_date"] = pd.to_datetime(dart_df["publish_date"], errors="coerce")
        valid = dart_df[dart_df["publish_date"].notna() & (dart_df["publish_date"] <= ref)]
        if valid.empty:
            continue
        pub_date   = valid.sort_values("publish_date").iloc[-1]["publish_date"]
        days_since = count_trading_days(pub_date, ref)
        if days_since <= 3 and ticker not in pending:
            wait_end = add_trading_days(pub_date, 2)
            pending[ticker] = wait_end.strftime("%Y-%m-%d")
            newly_announced.append(ticker)

    if newly_announced:
        logger.info("B 신규 공시 %d종목 → 3일 대기: %s", len(newly_announced), newly_announced[:5])

    completed = [t for t, we in list(pending.items()) if ref >= pd.Timestamp(we)]
    for t in completed:
        del pending[t]
    _save_pending(pending)

    if completed:
        logger.info("B 3일 완료 %d종목 재스코어링: %s", len(completed), completed[:5])
        _score_tickers_b(completed, ref)  # factor_dataset_B.parquet에 반영됨

    # watchlist(참고용) 갱신 — 매일 전체 유니버스 최신 저장값 기준으로 상위 30 재산출.
    # agent_c.py의 _get_top30_pool()과 동일한 소스(factor_dataset_B.parquet 최신 스냅샷)를
    # 사용하므로, 3일 대기 완료 종목이 없는 날에도 실제 매매 풀과 이 캐시가 항상 일치한다.
    try:
        all_scored = _score_from_saved_snapshot(universe, ref)
        new_top = _select_top_n(all_scored)
        wl["agent_b"]      = new_top
        wl["last_updated"] = ref_date
        _save_watchlist(wl)
        logger.info("Agent B Top %d 갱신: %s", POOL_SIZE, new_top[:5])
    except Exception as e:
        logger.warning("Agent B watchlist 갱신 실패(참고용이라 매매엔 영향 없음): %s", e)

    return wl.get("agent_b", [])


def _score_from_saved_snapshot(tickers: list[str], ref_date: pd.Timestamp) -> pd.DataFrame:
    """
    저장된 factor_dataset_B.parquet에서 각 종목의 가장 최근(ref_date 이전) 행을
    가져와 점수만 다시 매긴다 — 매번 전체를 재계산하지 않고 이미 저장된 값 재사용.
    """
    from config.settings import BACKTEST_DIR, AGENT_B_FEATURES
    path = BACKTEST_DIR / "results" / "factor_dataset_B.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "xgb_score"])

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] < ref_date) & (df["ticker"].isin(tickers))]
    if df.empty:
        return pd.DataFrame(columns=["ticker", "xgb_score"])
    latest = df.sort_values("date").groupby("ticker").tail(1).copy()

    model = load_model("B")
    if model is None:
        latest["xgb_score"] = 0.0
        return latest[["ticker", "xgb_score"]]

    missing = [c for c in AGENT_B_FEATURES if c not in latest.columns]
    for c in missing:
        latest[c] = np.nan
    latest["xgb_score"] = predict_scores(model, latest, "B")
    return latest[["ticker", "xgb_score"]]


def friday_rerank_agent_b(ref_date: str | None = None) -> list[str]:
    """
    금요일 15:30 실행. 코스피200 전체 종목 재계산 → factor_dataset_B.parquet 전체 갱신
    → Top 30 재확정. 3일 대기 미완료 종목(ear_3d 등 미완성)도 포함해서 계산하되,
    build_live_snapshot 내부 로직이 미완료 항목은 null로 자연스럽게 처리한다.
    """
    if ref_date is None:
        ref_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    ref = pd.Timestamp(ref_date)

    wl = _load_watchlist()

    try:
        universe = get_universe_by_date(ref_date)
    except RuntimeError as e:
        logger.error("유니버스 로드 실패: %s", e)
        return wl.get("agent_b", [])

    logger.info("Agent B 전체 재랭킹 시작 (%s, %d 종목)", ref_date, len(universe))
    scored = _score_tickers_b(universe, ref)  # factor_dataset_B.parquet 전체 갱신
    top_n  = _select_top_n(scored)

    wl["agent_b"]      = top_n
    wl["last_updated"] = ref_date
    _save_watchlist(wl)

    logger.info("Agent B Top %d 재확정: %s", POOL_SIZE, top_n)
    return top_n


def run_agent_b(ref_date: str | None = None) -> list[str]:
    """반기 첫 월요일 실행. KOSPI200 유니버스 리밸런싱 시점에 전체 재랭킹 겸용."""
    if ref_date is None:
        ref_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    logger.info("Agent B 반기 리밸런싱 시작: %s", ref_date)
    top_n = friday_rerank_agent_b(ref_date)

    if PORTFOLIO_PATH.exists():
        portfolio = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
        held      = list(portfolio.get("holdings", {}).keys())
        excluded  = [t for t in held if t not in top_n]
        if excluded:
            logger.info("B 새 watchlist 미포함 보유 종목 (자연 청산 대기): %s", excluded)

    return top_n
