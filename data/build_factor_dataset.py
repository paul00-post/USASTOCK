"""
팩터 데이터셋 빌더.

매주 금요일 스냅샷 기준으로 전 종목 팩터를 계산하여
backtest/results/factor_dataset_{A|B}.parquet 생성.

성능 최적화:
  - 시작 시 price_cache + dart_cache 전체를 메모리에 한 번만 로드
  - 반기 단위(유니버스 동일 기간) 내 풀 배열 재사용
  → 디스크 읽기: 23만 번 → 774번

생존편향 방지:
  - 각 시점의 역사적 KOSPI200 사용 (build_universe.py 반기 스냅샷)
  - publish_date <= ref_date 필터 (룩어헤드 바이어스 방지)

실행: python -m data.build_factor_dataset [A|B]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.settings import (
    AGENT_A_FACTOR_META,
    AGENT_A_FACTORS_BINARY,
    AGENT_A_FACTORS_TIMESERIES,
    AGENT_A_FEATURES,
    AGENT_B_FACTOR_META,
    AGENT_B_FACTORS_BINARY,
    AGENT_B_FACTORS_TIMESERIES,
    AGENT_B_FEATURES,
    AGENT_B_START_YEAR,
    BACKTEST_DIR,
    LABEL_CONFIG,
    PRICE_START_DATE,
    TIME_Z_QUARTERS_PREFERRED,
)
from data.build_universe import get_universe_by_date, get_all_tickers_until
from data.dart_collector import load_dart_cache
from data.build_price_cache import load_price_cache
from data.fetch_shares import load_shares
from data.sector_classifier import load_sector_map
from data.factor_engine import (
    add_ttm_columns,
    apply_feature_transform,
    compute_binary_a,
    compute_binary_b,
    compute_cagr,
    compute_ear_3d_fast,
    compute_sector_momentum_fast,
    extract_raw_factors_a,
    extract_raw_factors_b,
    _safe_div,
    _safe_float,
)
from utils.calendar_utils import (
    add_trading_days,
    count_trading_days,
    get_trading_days,
)
from utils.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR  = BACKTEST_DIR / "results"
SNAPSHOT_DIR = RESULTS_DIR / "weekly_snapshots"


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _get_all_fridays(start_year: int, end_year: int) -> list[pd.Timestamp]:
    """start_year 1월 ~ end_year 12월 말 사이의 모든 금요일(영업일) 목록."""
    start = pd.Timestamp(year=start_year, month=1, day=1)
    end   = pd.Timestamp(year=end_year,   month=12, day=31)
    days  = get_trading_days(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    return [d for d in days if d.dayofweek == 4]  # 4 = 금요일


def _pending_ear3d(publish_date: pd.Timestamp, ref_date: pd.Timestamp) -> bool:
    """ref_date 기준 3일 대기가 아직 완료되지 않은 종목 여부."""
    wait_end = add_trading_days(publish_date, 2)
    return ref_date < wait_end


def _semiannual_key(ref_date: pd.Timestamp) -> tuple[int, int]:
    """반기 키 — (year, 1) = 1~6월, (year, 7) = 7~12월."""
    return (ref_date.year, 1 if ref_date.month < 7 else 7)


# ── 경량 사전 로드 (실운용 — 현재 유니버스만) ────────────────────────────────

_KOSPI_PROXY_TICKER = "SPY"  # SPDR S&P 500 ETF — 시장중립화·섹터모멘텀 벤치마크


def _preload_universe(tickers: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    _preload_all()의 경량 버전 — 전체 역사적 유니버스가 아니라 지정된
    종목 목록(보통 현재 코스피200 약 200종목)만 메모리에 올린다.
    실운용 일별 재채점에서 사용 (수천 종목을 전부 로드하는 배치용 버전과 달리 몇 초 내로 끝남).

    SPY는 ETF라 S&P500 구성종목이 아니므로 tickers 목록에
    보통 없다 — 명시적으로 항상 포함시켜야 한다. 빠뜨리면 kospi_df=None이 되어
    _kospi_return_1m/_kospi_period_return이 조용히 0.0을 반환하고, 그 결과
    sector_momentum과 label_3m이 시장중립화 없이 원시 수익률 그대로 저장된다
    (2026-07-09 발견 — 지금까지의 label_3m 전부가 이 상태였음).
    """
    tickers_with_proxy = list(tickers)
    if _KOSPI_PROXY_TICKER not in tickers_with_proxy:
        tickers_with_proxy.append(_KOSPI_PROXY_TICKER)

    price_mem: dict[str, pd.DataFrame] = {}
    for ticker in tickers_with_proxy:
        df = load_price_cache(ticker)
        if df is not None and not df.empty:
            df = df.copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            price_mem[ticker] = df

    dart_mem: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = load_dart_cache(ticker)
        if df is not None and not df.empty:
            df = df.copy()
            df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
            df["report_date"]  = pd.to_datetime(df["report_date"],  errors="coerce")
            df = add_ttm_columns(df)
            dart_mem[ticker] = df

    return price_mem, dart_mem


def build_live_snapshot(agent: str, ref_date: pd.Timestamp, tickers: list[str] | None = None) -> pd.DataFrame:
    """
    실운용용 — 지정 시점 기준 전체 유니버스(기본: 현재 코스피200) 스냅샷을
    즉석에서 계산한다. 배치 백테스팅과 완전히 동일한 계산 로직(_build_pools,
    _build_row)을 재사용하므로 학습 시점과 실운용 시점의 피처 산출 방식이 어긋나지 않는다.

    반환된 DataFrame을 factor_dataset_{agent}.parquet에 append하면
    Agent C의 _get_top30_pool()이 곧바로 최신 데이터를 사용하게 된다.
    """
    if tickers is None:
        tickers = get_universe_by_date(ref_date.strftime("%Y-%m-%d"))

    sector_map = load_sector_map()
    shares_map = load_shares()
    price_mem, dart_mem = _preload_universe(tickers)
    kospi_df = price_mem.get("SPY")

    sector_pools, all_pools = _build_pools(
        tickers, ref_date, sector_map, shares_map, price_mem, dart_mem, agent
    )
    kospi_ret = _kospi_return_1m(kospi_df, ref_date)

    mktcap_map: dict[str, float] = {}
    ret_3m_list: list[float] = []
    ref_3m = ref_date - pd.Timedelta(days=90)
    for t in tickers:
        pdf = price_mem.get(t)
        sh  = shares_map.get(t)
        if pdf is None:
            continue
        p_now = pdf[pdf.index <= ref_date]
        if p_now.empty:
            continue
        p_close = float(p_now.iloc[-1]["Close"])
        if sh:
            mktcap_map[t] = p_close * sh
        p_3m = pdf[pdf.index <= ref_3m]
        if not p_3m.empty:
            ret_3m_list.append((p_close - p_3m.iloc[-1]["Close"]) / p_3m.iloc[-1]["Close"])
    bottom10 = float(np.percentile(ret_3m_list, 10)) if ret_3m_list else -0.15

    unique_sectors = sector_map[sector_map["ticker"].isin(tickers)]["custom_sector"].dropna().unique()
    sector_momentum_map: dict[str, float | None] = {}
    for sec in unique_sectors:
        sec_tickers = sector_map[sector_map["custom_sector"] == sec]["ticker"].tolist()
        sector_momentum_map[sec] = compute_sector_momentum_fast(
            sec_tickers, price_mem, ref_date, mktcap_map, kospi_ret,
        )

    rows = []
    for ticker in tickers:
        row = _build_row(
            ticker, ref_date, agent,
            sector_map, shares_map,
            price_mem, dart_mem,
            sector_pools, all_pools,
            sector_momentum_map, bottom10,
        )
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _save_snapshot_append(agent: str, snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    스냅샷 DataFrame을 factor_dataset_{agent}.parquet에 append한다.
    같은 (date, ticker) 조합이 이미 있으면 새 값으로 덮어쓴다 (같은 날 재실행 대비).
    agent_a.py/agent_b.py의 일별·금요일 재스코어링에서도 이 함수로 실제 파일을
    갱신해야 Agent C의 _get_top30_pool()이 신선한 데이터를 읽는다.
    """
    out_path = RESULTS_DIR / f"factor_dataset_{agent}.parquet"
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing["date"] = pd.to_datetime(existing["date"])
        key = existing["date"].astype(str) + "_" + existing["ticker"]
        new_key = snapshot["date"].astype(str) + "_" + snapshot["ticker"]
        existing = existing[~key.isin(set(new_key))]
        combined = pd.concat([existing, snapshot], ignore_index=True)
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        combined = snapshot

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_parquet(out_path, index=False)
    return combined


def append_live_snapshot(agent: str, ref_date: pd.Timestamp, tickers: list[str] | None = None) -> int:
    """
    build_live_snapshot() 결과를 factor_dataset_{agent}.parquet에 append한다.
    반환값: 저장된 종목 수
    """
    snapshot = build_live_snapshot(agent, ref_date, tickers)
    if snapshot.empty:
        logger.warning("Agent %s 라이브 스냅샷: 생성된 행 없음 (%s)", agent, ref_date.date())
        return 0

    combined = _save_snapshot_append(agent, snapshot)
    logger.info(
        "Agent %s 라이브 스냅샷 저장: %s (%d종목, 누적 %d행)",
        agent, ref_date.date(), len(snapshot), len(combined),
    )
    return len(snapshot)


# ── 메모리 사전 로드 (배치/백테스팅 전용 — 전체 역사적 유니버스) ─────────────

def _preload_all(end_year: int) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    price_cache + dart_cache 전체를 메모리에 로드.
    반환: (price_mem, dart_mem)
      price_mem[ticker] = Date-인덱스 DataFrame (Open/High/Low/Close/Volume)
      dart_mem[ticker]  = report_date/publish_date 기준 재무 DataFrame
    """
    all_tickers = get_all_tickers_until(end_year)
    logger.info("가격 캐시 메모리 로드 중... (%d 종목)", len(all_tickers))

    # SPY는 ETF라 get_all_tickers_until()의 S&P500 구성종목
    # 목록에 없다 — 명시적으로 추가하지 않으면 kospi_df=None이 되어 시장중립화가
    # 조용히 0으로 처리된다 (2026-07-09 발견, _preload_universe와 동일 문제).
    price_tickers = list(all_tickers)
    if _KOSPI_PROXY_TICKER not in price_tickers:
        price_tickers.append(_KOSPI_PROXY_TICKER)

    price_mem: dict[str, pd.DataFrame] = {}
    for ticker in price_tickers:
        df = load_price_cache(ticker)
        if df is not None and not df.empty:
            df = df.copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            price_mem[ticker] = df
    logger.info("가격 캐시 로드 완료: %d 종목", len(price_mem))

    logger.info("DART 캐시 메모리 로드 중... (%d 종목)", len(all_tickers))
    dart_mem: dict[str, pd.DataFrame] = {}
    for ticker in all_tickers:
        df = load_dart_cache(ticker)
        if df is not None and not df.empty:
            df = df.copy()
            df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
            df["report_date"]  = pd.to_datetime(df["report_date"],  errors="coerce")
            df = add_ttm_columns(df)
            dart_mem[ticker] = df
    logger.info("DART 캐시 로드 완료: %d 종목", len(dart_mem))

    return price_mem, dart_mem


# ── 코스피200 수익률 (메모리 버전) ────────────────────────────────────────────

def _kospi_return_1m(kospi_df: pd.DataFrame, ref_date: pd.Timestamp) -> float:
    """KOSPI200 ETF 1개월 수익률 (섹터 모멘텀 기준)."""
    if kospi_df is None or kospi_df.empty:
        return 0.0
    now_row  = kospi_df[kospi_df.index <= ref_date]
    past_row = kospi_df[kospi_df.index <= ref_date - pd.Timedelta(days=30)]
    if now_row.empty or past_row.empty:
        return 0.0
    return float((now_row.iloc[-1]["Close"] - past_row.iloc[-1]["Close"]) / past_row.iloc[-1]["Close"])


def _get_kospi200_return_1m(ref_date: pd.Timestamp) -> float:
    """
    KOSPI200 ETF(SPY) 1개월 수익률 — 단일 조회용 래퍼.

    agent_a.py/agent_b.py의 실운용 일별 부분 재스코어링에서 사용한다
    (배치 처리 시에는 _kospi_return_1m()에 미리 로드해둔 kospi_df를 직접 전달).
    """
    kospi_df = load_price_cache("SPY")
    if kospi_df is None or kospi_df.empty:
        return 0.0
    kospi_df = kospi_df.set_index("Date").sort_index()
    return _kospi_return_1m(kospi_df, pd.Timestamp(ref_date))


def _kospi_period_return(
    kospi_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """KOSPI200 ETF start→end 구간 수익률 (라벨 시장중립화)."""
    if kospi_df is None or kospi_df.empty:
        return 0.0
    start_row = kospi_df[kospi_df.index >= start_date].head(1)
    end_row   = kospi_df[kospi_df.index <= end_date]
    if start_row.empty or end_row.empty:
        return 0.0
    p_start = float(start_row.iloc[0]["Open"])
    if p_start == 0:
        return 0.0
    return float((end_row.iloc[-1]["Close"] - p_start) / p_start)


# ── 풀 배열 계산 (메모리 버전) ────────────────────────────────────────────────

def _build_pools(
    tickers: list[str],
    ref_date: pd.Timestamp,
    sector_map: pd.DataFrame,
    shares_map: dict[str, int],
    price_mem: dict[str, pd.DataFrame],
    dart_mem: dict[str, pd.DataFrame],
    agent: str,
) -> tuple[dict, dict]:
    """
    ref_date 기준 섹터별·전체 팩터 풀 배열 계산.
    반환: (sector_pools, all_pools)
    """
    factor_meta = AGENT_A_FACTOR_META if agent == "A" else AGENT_B_FACTOR_META

    raw_all: dict[str, dict] = {}
    for ticker in tickers:
        price_df = price_mem.get(ticker)
        dart_df  = dart_mem.get(ticker)
        shares   = shares_map.get(ticker)
        if price_df is None or dart_df is None or not shares:
            continue
        p_row = price_df[price_df.index <= ref_date]   # 금요일 종가 포함
        if p_row.empty:
            continue
        price = float(p_row.iloc[-1]["Close"])
        valid = dart_df[dart_df["publish_date"].notna() & (dart_df["publish_date"] <= ref_date)]
        if valid.empty:
            continue
        valid_pub = valid.sort_values("publish_date")
        valid_rep = valid.sort_values("report_date")
        fin      = valid_pub.iloc[-1]
        fin_prev = valid_pub.iloc[-5] if len(valid) >= 5 else None

        if agent == "A":
            raw = extract_raw_factors_a(fin, price, shares, valid, valid_rep)
        else:
            raw = extract_raw_factors_b(
                fin, fin_prev, price, shares, valid,
                price_df,    # Date-인덱스 그대로
                ref_date,
                valid_rep,   # report_date 정렬 버전
            )
        raw_all[ticker] = raw

    # 전체 풀
    all_pools: dict[str, np.ndarray] = {}
    for factor_name, _, _ in factor_meta:
        vals = [
            v[factor_name] for v in raw_all.values()
            if v.get(factor_name) is not None and not np.isnan(v[factor_name])
        ]
        all_pools[factor_name] = np.array(vals, dtype=float)

    # 섹터별 풀
    sectors = sector_map["custom_sector"].dropna().unique()
    sector_pools: dict[str, dict[str, np.ndarray]] = {}
    for sec in sectors:
        sec_tickers = sector_map[sector_map["custom_sector"] == sec]["ticker"].tolist()
        sector_pools[sec] = {}
        for factor_name, _, _ in factor_meta:
            vals = [
                raw_all[t][factor_name] for t in sec_tickers
                if t in raw_all
                and raw_all[t].get(factor_name) is not None
                and not np.isnan(raw_all[t][factor_name])
            ]
            sector_pools[sec][factor_name] = np.array(vals, dtype=float)

    return sector_pools, all_pools


# ── 로더 팩토리 ──────────────────────────────────────────────────────────────

def _make_price_loader(price_mem: dict[str, pd.DataFrame]):
    """
    price_mem(Date-인덱스 기반)을 price_loader 형식으로 변환하는 콜백 생성.
    factor_engine의 compute_* 함수는 'Date' 컬럼 DataFrame을 기대한다.
    """
    def loader(ticker: str) -> pd.DataFrame | None:
        df = price_mem.get(ticker)
        if df is None:
            return None
        return df.reset_index()   # "Date" 인덱스 → "Date" 컬럼
    return loader


# ── time_z 이력 배열 계산 ────────────────────────────────────────────────────

def _build_factor_history(
    valid_dart: pd.DataFrame,
    shares: int,
    agent: str,
    price_df: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """
    valid_dart 과거 기록에서 time_z용 팩터 이력 배열 계산.
    현재 분기(마지막 publish_date 행)를 제외한 직전 최대 8분기.

    price_df가 주어지면(Agent B) 가격 기반 팩터(peg, psr, cash_to_mktcap,
    momentum_3m, momentum_6m, high52w_pct)의 이력도 각 과거 분기 시점의
    실제 주가를 조회해 계산한다 — price_df가 없으면 해당 팩터는 생략(NaN).
    """
    from data.factor_engine import compute_roic

    df   = valid_dart.sort_values("publish_date")
    hist_full = df.iloc[:-1]  # 현재 분기 제외 (성장률 lookback용 — 8개로 자르기 전)
    hist = hist_full.tail(TIME_Z_QUARTERS_PREFERRED) if len(hist_full) > TIME_Z_QUARTERS_PREFERRED else hist_full
    if hist.empty:
        return {}

    def _col(col: str, frame: pd.DataFrame = hist) -> np.ndarray:
        if col not in frame.columns:
            return np.full(len(frame), np.nan)
        return pd.to_numeric(frame[col], errors="coerce").values.astype(float)

    def _ratio(num: str, den: str) -> np.ndarray:
        n = _col(num)
        d = _col(den)
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = np.where((~np.isnan(d)) & (d != 0), n / d, np.nan)
        return vals.astype(float)

    def _roic_hist() -> np.ndarray:
        # iterrows 제거 → 완전 벡터화
        # op_profit/tax_expense는 TTM(최근 12개월) 기준 — extract_raw_factors_b의
        # compute_roic()와 동일 척도로 맞춰야 "현재값 vs 과거값" 비교(time_z)가 일관됨.
        op_p   = _col("op_profit_ttm")
        tax    = np.where(np.isnan(_col("tax_expense_ttm")), 0.0, _col("tax_expense_ttm"))
        eq     = _col("total_equity")
        debt   = np.where(np.isnan(_col("total_debt")), 0.0, _col("total_debt"))
        cash_v = np.where(np.isnan(_col("cash")), 0.0, _col("cash"))
        with np.errstate(invalid="ignore", divide="ignore"):
            tax_rate = np.where(op_p != 0, tax / (op_p + 1e-9), 0.0)
            nopat    = op_p * (1.0 - tax_rate)
            inv_cap  = eq + debt - cash_v
            result   = np.where(
                (~np.isnan(op_p)) & (~np.isnan(eq)) & (inv_cap > 0),
                nopat / inv_cap,
                np.nan,
            )
        return result.astype(float)

    # ── YoY 성장률 이력 (4분기 전 대비) — 8개로 자르기 전 전체 이력에서 계산한 뒤
    #    hist(마지막 8개)와 같은 길이로 뒤에서부터 잘라 정렬한다.
    #    (lookback이 4분기 전 값을 참조해야 하므로 자르기 전 데이터가 필요함)
    def _yoy_growth_full(col: str, divisor: float | None = None) -> np.ndarray:
        if col not in hist_full.columns:
            return np.full(len(hist), np.nan)
        vals = pd.to_numeric(hist_full[col], errors="coerce").values.astype(float)
        if divisor:
            vals = vals / divisor
        prev = np.roll(vals, 4)
        prev[:4] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            growth = np.where((~np.isnan(prev)) & (prev != 0), (vals - prev) / np.abs(prev), np.nan)
        return growth[-len(hist):] if len(hist) > 0 else growth

    if agent == "A":
        return {
            "roe":               _ratio("net_income",        "total_equity"),
            "roa":               _ratio("net_income",        "total_assets"),
            "roic":              _roic_hist(),
            "net_margin":        _ratio("net_income",        "revenue"),
            "op_margin":         _ratio("op_profit",         "revenue"),
            "asset_turnover":    _ratio("revenue",           "total_assets"),
            "debt_ratio":        _ratio("total_liabilities", "total_assets"),
            "interest_coverage": np.where(
                pd.to_numeric(hist["interest_expense"], errors="coerce").values > 0,
                pd.to_numeric(hist["op_profit"],       errors="coerce").values /
                pd.to_numeric(hist["interest_expense"], errors="coerce").values,
                np.nan,
            ).astype(float),
            "current_ratio":     _ratio("current_assets",   "current_liabilities"),
            "fcf_margin":        _ratio("cfo",              "revenue"),
            "cash_ratio":        _ratio("cash",             "current_liabilities"),
        }
    else:  # B
        revenue_growth_hist   = _yoy_growth_full("revenue")
        op_profit_growth_hist = _yoy_growth_full("op_profit")
        eps_growth_hist       = _yoy_growth_full("net_income", divisor=shares if shares else None)

        # 매출증가율이 0에 가까우면(_OP_LEVERAGE_MIN_DENOM 미만) 나눗셈 결과가
        # 실제 신호와 무관한 극단값으로 튀므로 생략 — 현재값 계산과 동일 기준.
        from data.factor_engine import _OP_LEVERAGE_MIN_DENOM
        with np.errstate(invalid="ignore", divide="ignore"):
            operating_leverage_hist = np.where(
                (~np.isnan(revenue_growth_hist)) & (np.abs(revenue_growth_hist) >= _OP_LEVERAGE_MIN_DENOM),
                op_profit_growth_hist / revenue_growth_hist,
                np.nan,
            ).astype(float)

        # ── 5년 CAGR 이력 — report_date 순서상 해당 시점까지의 데이터만 사용해
        #    그 시점 기준 5년 CAGR을 재계산 (매 분기 "그때 기준 CAGR"을 구하는 것).
        # revenue_ttm/net_income_ttm 기준 — extract_raw_factors_b의 rev_cagr/eps_cagr와
        # 동일 척도로 맞춤 (원본 분기값은 분기 유형이 어긋나면 왜곡될 수 있음).
        df_by_rep = valid_dart.sort_values("report_date")
        rep_pos = {idx: pos for pos, idx in enumerate(df_by_rep.index)}
        if shares and "net_income_ttm" in df_by_rep.columns:
            df_by_rep = df_by_rep.copy()
            df_by_rep["_eps_col"] = pd.to_numeric(df_by_rep["net_income_ttm"], errors="coerce") / shares

        def _cagr_hist(col: str) -> np.ndarray:
            out = np.full(len(hist), np.nan)
            if col not in df_by_rep.columns:
                return out
            for i, idx in enumerate(hist.index):
                pos = rep_pos.get(idx)
                if pos is None:
                    continue
                val = compute_cagr(df_by_rep.iloc[:pos + 1], col, 5, already_sorted=True)
                if val is not None:
                    out[i] = val
            return out

        revenue_cagr_hist = _cagr_hist("revenue_ttm")
        eps_cagr_hist     = _cagr_hist("_eps_col") if shares else np.full(len(hist), np.nan)

        # ── 가격 기반 팩터 이력 — 각 과거 분기 시점의 실제 주가로 재계산 ──────
        n = len(hist)
        mom3 = np.full(n, np.nan); mom6 = np.full(n, np.nan); h52 = np.full(n, np.nan)
        psr_h = np.full(n, np.nan); c2m_h = np.full(n, np.nan); peg_h = np.full(n, np.nan)

        if price_df is not None and not price_df.empty and shares:
            for i, (_, row) in enumerate(hist.iterrows()):
                pub = row.get("publish_date")
                if pd.isna(pub):
                    continue
                pub = pd.Timestamp(pub)
                p_now_row = price_df[price_df.index <= pub]
                if p_now_row.empty:
                    continue
                p_now = float(p_now_row.iloc[-1]["Close"])

                p_3m_row = price_df[price_df.index <= pub - pd.Timedelta(days=90)]
                if not p_3m_row.empty:
                    base = float(p_3m_row.iloc[-1]["Close"])
                    if base:
                        mom3[i] = (p_now - base) / base

                p_6m_row = price_df[price_df.index <= pub - pd.Timedelta(days=180)]
                if not p_6m_row.empty:
                    base = float(p_6m_row.iloc[-1]["Close"])
                    if base:
                        mom6[i] = (p_now - base) / base

                p_1y = price_df[(price_df.index >= pub - pd.Timedelta(days=365)) & (price_df.index <= pub)]
                if not p_1y.empty:
                    hi, lo = float(p_1y["High"].max()), float(p_1y["Low"].min())
                    if hi != lo:
                        h52[i] = (p_now - lo) / (hi - lo)

                revenue_ttm_v = _safe_float(row, "revenue_ttm")
                net_inc_ttm_v = _safe_float(row, "net_income_ttm")
                cash_v  = _safe_float(row, "cash")
                mktcap  = p_now * shares

                if revenue_ttm_v:
                    psr_h[i] = _safe_div(p_now, _safe_div(revenue_ttm_v, shares)) or np.nan
                if mktcap and cash_v is not None:
                    c2m_h[i] = cash_v / mktcap

                eps_val = _safe_div(net_inc_ttm_v, shares) if net_inc_ttm_v is not None else None
                per_val = (p_now / eps_val) if (eps_val and eps_val > 0) else None
                eg = eps_growth_hist[i] if i < len(eps_growth_hist) else np.nan
                if per_val is not None and not np.isnan(eg) and eg > 0:
                    peg_h[i] = per_val / (eg * 100)

        return {
            "roic":               _roic_hist(),
            "gross_margin_trend": _ratio("gross_profit", "revenue"),
            "rd_ratio":           _ratio("rd_expense",   "revenue"),
            "revenue_growth":     revenue_growth_hist,
            "op_profit_growth":   op_profit_growth_hist,
            "eps_growth":         eps_growth_hist,
            "operating_leverage": operating_leverage_hist,
            "revenue_cagr_5y":    revenue_cagr_hist,
            "eps_cagr_5y":        eps_cagr_hist,
            "momentum_3m":        mom3,
            "momentum_6m":        mom6,
            "high52w_pct":        h52,
            "psr":                psr_h,
            "cash_to_mktcap":     c2m_h,
            "peg":                peg_h,
        }


# ── 스냅샷 행 생성 (메모리 버전) ──────────────────────────────────────────────

def _build_row(
    ticker: str,
    ref_date: pd.Timestamp,
    agent: str,
    sector_map: pd.DataFrame,
    shares_map: dict[str, int],
    price_mem: dict[str, pd.DataFrame],
    dart_mem: dict[str, pd.DataFrame],
    sector_pools: dict,
    all_pools: dict,
    sector_momentum_map: dict[str, float | None],   # 섹터별 사전 계산된 모멘텀
    bottom10_ret3m: float,
) -> dict | None:
    """단일 종목 × 단일 날짜 스냅샷 행 생성 (메모리 기반)."""
    price_df = price_mem.get(ticker)
    if price_df is None:
        return None
    p_row = price_df[price_df.index <= ref_date]   # 금요일 종가 포함
    if p_row.empty:
        return None
    price = float(p_row.iloc[-1]["Close"])

    shares = shares_map.get(ticker)
    if not shares:
        return None

    dart_df = dart_mem.get(ticker)
    if dart_df is None:
        return None
    valid_dart = dart_df[dart_df["publish_date"].notna() & (dart_df["publish_date"] <= ref_date)]
    if valid_dart.empty:
        return None
    valid_dart   = valid_dart.sort_values("publish_date")   # 한 번만 정렬
    fin          = valid_dart.iloc[-1]
    publish_date = fin["publish_date"]

    # CAGR 계산용: report_date 기준 정렬 (compute_cagr 내부 중복 정렬 방지)
    valid_dart_by_rep = valid_dart.sort_values("report_date")

    # 섹터
    sector_row    = sector_map[sector_map["ticker"] == ticker]
    custom_sector = sector_row.iloc[0]["custom_sector"] if not sector_row.empty else None
    if custom_sector is None or (isinstance(custom_sector, float) and np.isnan(custom_sector)):
        return None

    fin_prev_year = valid_dart.iloc[-5] if len(valid_dart) >= 5 else None

    days_since = count_trading_days(publish_date, ref_date)

    # ear_3d (fast 버전: price_mem 직접 사용 → copy/set_index 없음)
    ear_final = ear_trend = None
    if not _pending_ear3d(publish_date, ref_date):
        ear_final, ear_trend = compute_ear_3d_fast(ticker, publish_date, price_mem)

    # sector_momentum: 이미 섹터별 사전 계산됨
    sec_momentum = sector_momentum_map.get(custom_sector)

    timeseries = {
        "sector_momentum":     sec_momentum,
        "days_since_earnings": days_since,
        "ear_3d_final":        ear_final,
        "ear_3d_trend":        ear_trend,
    }

    sec_p = sector_pools.get(custom_sector, {})

    if agent == "A":
        raw = extract_raw_factors_a(fin, price, shares, valid_dart, valid_dart_by_rep)

        # roe_5y_avg: 직접 raw 컬럼으로 계산 (publish_date <= ref_date 기록만 사용)
        roe_hist = valid_dart.sort_values("publish_date").tail(20)
        if "net_income" in roe_hist.columns and "total_equity" in roe_hist.columns:
            ni_vals  = pd.to_numeric(roe_hist["net_income"],   errors="coerce").values
            eq_vals  = pd.to_numeric(roe_hist["total_equity"], errors="coerce").values
            with np.errstate(invalid="ignore", divide="ignore"):
                roe_vals = np.where((~np.isnan(eq_vals)) & (eq_vals != 0), ni_vals / eq_vals, np.nan)
            valid_roe = roe_vals[~np.isnan(roe_vals)]
            if len(valid_roe) > 0:
                raw["roe_5y_avg"] = float(np.mean(valid_roe))

        if fin_prev_year is not None:
            rev_now  = _safe_float(fin, "revenue")
            rev_prev = _safe_float(fin_prev_year, "revenue")
            if rev_now and rev_prev and rev_prev != 0:
                raw["revenue_growth"] = (rev_now - rev_prev) / abs(rev_prev)

        history_pools = _build_factor_history(valid_dart, shares, "A")
        features = apply_feature_transform(raw, AGENT_A_FACTOR_META, sec_p, all_pools, history_pools)

        p_now = price_df[price_df.index <= ref_date]
        p_3m  = price_df[price_df.index <= ref_date - pd.Timedelta(days=90)]
        ret_3m = None
        if not p_now.empty and not p_3m.empty:
            ret_3m = (p_now.iloc[-1]["Close"] - p_3m.iloc[-1]["Close"]) / p_3m.iloc[-1]["Close"]

        binary = compute_binary_a(
            valid_dart,
            cfo_current=_safe_float(fin, "cfo"),
            equity_current=_safe_float(fin, "total_equity"),
            return_3m=ret_3m,
            return_3m_bottom10pct=bottom10_ret3m,
        )
        features.update(binary)
        features.update(timeseries)
        feature_cols = AGENT_A_FEATURES

    else:  # B
        raw = extract_raw_factors_b(
            fin, fin_prev_year, price, shares, valid_dart,
            price_df,           # Date-인덱스 그대로 (reset_index 오버헤드 제거)
            ref_date,
            valid_dart_by_rep,  # report_date 정렬 버전
        )

        history_pools = _build_factor_history(valid_dart, shares, "B", price_df=price_df)
        features = apply_feature_transform(raw, AGENT_B_FACTOR_META, sec_p, all_pools, history_pools)

        binary = compute_binary_b(valid_dart, equity_current=_safe_float(fin, "total_equity"))
        features.update(binary)
        features.update(timeseries)
        feature_cols = AGENT_B_FEATURES

    row = {
        "date":          ref_date.strftime("%Y-%m-%d"),
        "ticker":        ticker,
        "custom_sector": custom_sector,
        "publish_date":  publish_date.strftime("%Y-%m-%d") if pd.notna(publish_date) else None,
        "label_3m":      None,
    }
    for col in feature_cols:
        row[col] = features.get(col, np.nan)
    return row


# ── 메인 빌드 함수 ────────────────────────────────────────────────────────────

def build_factor_dataset(agent: str = "A") -> pd.DataFrame:
    """
    전 기간 팩터 데이터셋 생성.
    agent = "A" or "B"
    """
    assert agent in ("A", "B"), "agent는 'A' 또는 'B'"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"factor_dataset_{agent}.parquet"

    start_year = AGENT_B_START_YEAR   # 2009
    end_year   = pd.Timestamp.today().year
    fridays    = _get_all_fridays(start_year, end_year)

    sector_map = load_sector_map()
    shares_map = load_shares()

    # ── 전체 데이터 메모리 사전 로드 ──────────────────────────────────────────
    price_mem, dart_mem = _preload_all(end_year)
    kospi_df = price_mem.get("SPY")   # SPDR S&P 500 ETF (벤치마크)

    all_rows: list[dict] = []

    # ── 반기 풀 캐시 (반기 경계에서만 재계산 → 속도 50× 향상) ─────────────────
    # 정확도 영향: cross_z/percentile 기준이 반기 초 DART 기준
    # 각 종목 행(valid_dart)은 여전히 해당 금요일 기준 → 팩터 값 정확
    _pool_sa_key: tuple | None = None
    sector_pools: dict = {}
    all_pools: dict = {}
    cached_tickers: list[str] = []

    for ref_date in tqdm(fridays, desc=f"Agent {agent} 팩터 빌드"):
        try:
            tickers = get_universe_by_date(ref_date.strftime("%Y-%m-%d"))
        except RuntimeError as e:
            logger.warning("유니버스 없음 (%s): %s", ref_date.date(), e)
            continue

        # 반기 경계 또는 유니버스 변경 시에만 풀 재계산
        sa_key = _semiannual_key(ref_date)
        if sa_key != _pool_sa_key or tickers != cached_tickers:
            sector_pools, all_pools = _build_pools(
                tickers, ref_date, sector_map, shares_map, price_mem, dart_mem, agent
            )
            _pool_sa_key   = sa_key
            cached_tickers = tickers
            logger.debug("풀 재계산: %s 반기 %s (%d 종목)", sa_key, ref_date.date(), len(tickers))

        # ── KOSPI200 수익률 ───────────────────────────────────────────────────
        kospi_ret = _kospi_return_1m(kospi_df, ref_date)

        # ── 시가총액 맵 (섹터 모멘텀 가중치) ─────────────────────────────────
        mktcap_map: dict[str, float] = {}
        ret_3m_list: list[float] = []
        ref_3m = ref_date - pd.Timedelta(days=90)
        for t in tickers:
            pdf = price_mem.get(t)
            sh  = shares_map.get(t)
            if pdf is None:
                continue
            p_now = pdf[pdf.index <= ref_date]
            if p_now.empty:
                continue
            p_close = float(p_now.iloc[-1]["Close"])
            if sh:
                mktcap_map[t] = p_close * sh
            p_3m = pdf[pdf.index <= ref_3m]
            if not p_3m.empty:
                ret_3m_list.append((p_close - p_3m.iloc[-1]["Close"]) / p_3m.iloc[-1]["Close"])
        bottom10 = float(np.percentile(ret_3m_list, 10)) if ret_3m_list else -0.15

        # ── 섹터 모멘텀: 섹터당 1회 계산 (티커당 계산 → 12-16배 절감) ─────────
        unique_sectors = sector_map[sector_map["ticker"].isin(tickers)]["custom_sector"].dropna().unique()
        sector_momentum_map: dict[str, float | None] = {}
        for sec in unique_sectors:
            sec_tickers = sector_map[sector_map["custom_sector"] == sec]["ticker"].tolist()
            sector_momentum_map[sec] = compute_sector_momentum_fast(
                sec_tickers, price_mem, ref_date, mktcap_map, kospi_ret,
            )

        # ── 3개월 수익률 하위 10% (no_value_trap 기준) 이미 위에서 계산 ──────

        # ── 종목별 행 생성 ────────────────────────────────────────────────────
        for ticker in tickers:
            row = _build_row(
                ticker, ref_date, agent,
                sector_map, shares_map,
                price_mem, dart_mem,
                sector_pools, all_pools,
                sector_momentum_map, bottom10,
            )
            if row is not None:
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if df.empty:
        logger.warning("Agent %s: 생성된 행 없음", agent)
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    logger.info("Agent %s 팩터 데이터셋 저장: %s (%d행)", agent, out_path, len(df))
    return df


# ── 라벨 사후 채움 ────────────────────────────────────────────────────────────

def fill_labels(agent: str = "A") -> None:
    """
    T+65 이후 경과한 스냅샷의 label_3m 사후 채움.

    label_3m = (T+61~T+65 5일 평균가 - T+1 시가) / T+1 시가 - KOSPI200_return
    """
    path = RESULTS_DIR / f"factor_dataset_{agent}.parquet"
    if not path.exists():
        logger.warning("factor_dataset_%s.parquet 없음", agent)
        return

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    # 라벨 계산에 필요한 price_cache 메모리 로드
    end_year = pd.Timestamp.today().year
    price_mem, _ = _preload_all(end_year)
    kospi_df = price_mem.get("SPY")

    today   = pd.Timestamp.today()
    updated = 0
    cutoff_bd = LABEL_CONFIG["label_cutoff_days"]  # 65

    for idx, row in df[df["label_3m"].isna()].iterrows():
        ref_date = row["date"]
        if count_trading_days(ref_date, today) < cutoff_bd:
            continue

        ticker   = row["ticker"]
        price_df = price_mem.get(ticker)
        if price_df is None:
            continue

        t1 = add_trading_days(ref_date, 1)
        t1_row = price_df[price_df.index >= t1].head(1)
        if t1_row.empty:
            continue
        entry_open = float(t1_row.iloc[0]["Open"])
        if entry_open == 0:
            continue

        avg_prices = []
        for offset in range(61, 66):
            dt    = add_trading_days(ref_date, offset)
            row_p = price_df[price_df.index <= dt]
            if not row_p.empty:
                avg_prices.append(float(row_p.iloc[-1]["Close"]))
        if len(avg_prices) < 3:
            continue
        price_avg = np.mean(avg_prices)

        t63       = add_trading_days(ref_date, 63)
        kospi_ret = _kospi_period_return(kospi_df, t1, t63)

        df.at[idx, "label_3m"] = (price_avg - entry_open) / entry_open - kospi_ret
        updated += 1

    df.to_parquet(path, index=False)
    logger.info("label_3m 채움 완료: %d행 업데이트 (Agent %s)", updated, agent)


if __name__ == "__main__":
    agent_arg = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
    assert agent_arg in ("A", "B"), "Usage: python -m data.build_factor_dataset [A|B]"
    build_factor_dataset(agent_arg)
    fill_labels(agent_arg)
