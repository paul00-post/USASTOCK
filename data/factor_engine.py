"""
팩터 계산 엔진.

각 재무 팩터 → {factor}_percentile, {factor}_cross_z, {factor}_time_z 3개 피처로 변환.
이진 팩터 → 0/1 그대로.
시계열 팩터 → 연속형 그대로.

설계 원칙:
  - 원본값(ROE=15.3% 등)은 피처에서 제외 (섹터 간 스케일 불일치 방지)
  - 이중 정규화 금지 (이미 percentile·Z-score로 변환됨)
  - XGBoost에 추가 정규화 없이 직접 투입
  - 섹터 종목 수 < 5개이면 KOSPI200 전체 기준으로 자동 폴백
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config.settings import (
    AGENT_A_FACTOR_META,
    AGENT_A_FACTORS_BINARY,
    AGENT_B_FACTOR_META,
    AGENT_B_FACTORS_BINARY,
    TIME_Z_QUARTERS_MIN,
    TIME_Z_QUARTERS_PREFERRED,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_SECTOR_MIN_COUNT = 5  # 미만이면 전체 풀로 폴백

# operating_leverage = 영업이익증가율 / 매출증가율 계산 시, 매출증가율이 0에
# 가까우면(거의 변동 없는 분기) 나눗셈 결과가 실제 신호와 무관하게 극단값으로
# 튄다(2026-07-09 HMM 2020Q1 사례: 매출증가율 -0.2% → 레버리지 -457).
# 매출증가율 절대값이 이 값 미만이면 계산 자체를 생략(None)해 극단값이
# cross_z/percentile 풀 전체를 왜곡하지 않게 한다.
_OP_LEVERAGE_MIN_DENOM = 0.01


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _safe_float(s: pd.Series | dict, col: str) -> float | None:
    """시리즈/딕셔너리에서 안전하게 float 추출. 0.0도 유효값으로 처리."""
    v = s.get(col) if isinstance(s, dict) else (s.get(col) if hasattr(s, "get") else None)
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


# ── 피처 변환 (단일 팩터 → 3개) ──────────────────────────────────────────────

def _safe_percentile(value: float, pool: np.ndarray, higher_is_better: bool) -> float:
    clean = pool[~np.isnan(pool)]
    if len(clean) == 0:
        return np.nan
    merged = np.append(clean, value)
    ranks  = rankdata(merged, method="average")
    pct    = ranks[-1] / len(merged)
    return pct if higher_is_better else (1.0 - pct)


def _safe_cross_z(value: float, pool: np.ndarray, higher_is_better: bool) -> float:
    clean = pool[~np.isnan(pool)]
    if len(clean) < 2:
        return np.nan
    mu, sigma = clean.mean(), clean.std(ddof=0)
    if sigma == 0:
        return 0.0
    sign = 1 if higher_is_better else -1
    return sign * (value - mu) / sigma


def _safe_time_z(value: float, history: np.ndarray, higher_is_better: bool) -> float:
    hist = history[~np.isnan(history)]
    if len(hist) < TIME_Z_QUARTERS_MIN:
        return np.nan
    hist = hist[-TIME_Z_QUARTERS_PREFERRED:]
    mu, sigma = hist.mean(), hist.std(ddof=0)
    if sigma == 0:
        return 0.0
    sign = 1 if higher_is_better else -1
    return sign * (value - mu) / sigma


def generate_continuous_features(
    factor_name: str,
    value: float | None,
    higher_is_better: bool,
    sector_values: np.ndarray,
    all_values: np.ndarray,
    own_history: np.ndarray,
    scope: str = "sector",
) -> dict[str, float]:
    """
    단일 연속형 팩터 → {factor_name}_percentile, _cross_z, _time_z 3개 피처.

    Parameters
    ----------
    sector_values : 동일 섹터 내 모든 종목의 현재 값 (자신 포함)
    all_values    : KOSPI200 전체 현재 값 (자신 포함)
    own_history   : 해당 종목 과거 N분기 값 (시계열Z 계산용)
    scope         : "sector" 또는 "all"
    """
    nan_result = {
        f"{factor_name}_percentile": np.nan,
        f"{factor_name}_cross_z":    np.nan,
        f"{factor_name}_time_z":     np.nan,
    }
    if value is None or np.isnan(value):
        return nan_result

    # 섹터 풀 < 5개 → 전체로 폴백
    sector_clean_count = int(np.sum(~np.isnan(sector_values)))
    pool = (
        sector_values
        if (scope == "sector" and sector_clean_count >= _SECTOR_MIN_COUNT)
        else all_values
    )

    return {
        f"{factor_name}_percentile": _safe_percentile(value, pool, higher_is_better),
        f"{factor_name}_cross_z":    _safe_cross_z(value, pool, higher_is_better),
        f"{factor_name}_time_z":     _safe_time_z(value, own_history, higher_is_better),
    }


# ── TTM(최근 12개월 합산) 계산 ─────────────────────────────────────────────────
#
# 배경: 한국 DART 공시 체계에는 별도 "4분기보고서"가 없다 — 연간(사업)보고서가
# 12개월 전체 총액을 그대로 보고한다. 그 결과 revenue/op_profit/net_income 같은
# 유량(flow) 항목의 원본값을 그대로 쓰는 팩터(roic, peg, psr, revenue_cagr_5y,
# eps_cagr_5y)는 "가장 최근 공시가 연간보고서인 회사"만 유독 3~4배 부풀려진
# 값으로 스냅샷에 찍히는 문제가 있었다.
#
# "연간총액 - (1~3분기 합)"으로 4분기 단독값을 역산하는 방식을 시도했으나,
# 분기 간 재작성(사업부 매각·중단사업 재분류·전기오류 수정)으로 연간총액이
# 1~3분기 합과 정확히 일치하지 않는 경우가 있어 매출액이 마이너스로 나오는 등
# 명백히 잘못된 값을 새로 만들어냈다 (2026-07-10 발견, 23건의 마이너스 매출).
#
# 대안: 4분기만 따로 떼어내려 하지 않고, 애초에 "최근 12개월 누적(TTM)"
# 기준으로 팩터를 재정의한다. 연간보고서 시점 = 그 자체가 이미 TTM이므로
# 뺄셈 없이 그대로 사용. 이후 분기는 직전 TTM에서 "작년 동일분기"만 빼고
# "올해 동일분기"만 더하는 한 분기 단위 롤링이라, 3개 분기를 한꺼번에 빼서
# 오차가 누적되는 문제가 구조적으로 없다.
def compute_ttm_series(history_sorted: pd.DataFrame, col: str) -> pd.Series:
    """
    report_date 오름차순 정렬된 dart_history에서 TTM(최근 12개월 합산) 시계열 계산.

    Q4(연간보고서) 행 = 연간총액을 그대로 TTM으로 사용 (뺄셈 불필요).
    Q1/Q2/Q3 행       = 직전 TTM - 작년 동일분기 단독값 + 올해 동일분기 단독값.
    직전 TTM 또는 작년 동일분기 값이 아직 없으면(초기 몇 년) NaN.
    """
    ttm = pd.Series(np.nan, index=history_sorted.index, dtype=float)
    if col not in history_sorted.columns:
        return ttm

    last_seen: dict[str, dict[int, float]] = {}
    prev_ttm: float | None = None

    for idx, row in history_sorted.iterrows():
        q    = row.get("quarter")
        year = row.get("bsns_year")
        val  = row.get(col)
        if pd.isna(val) or q is None or pd.isna(year):
            ttm.at[idx] = prev_ttm if q != "Q4" else np.nan
            continue

        val = float(val)
        year = int(year)

        if q == "Q4":
            prev_ttm = val  # 연간총액 = TTM 그 자체
        else:
            prior_val = last_seen.get(q, {}).get(year - 1)
            if prev_ttm is not None and prior_val is not None:
                prev_ttm = prev_ttm - prior_val + val
            else:
                prev_ttm = None  # 직전 TTM 또는 작년 동일분기 값 없음 — 롤링 불가
            last_seen.setdefault(q, {})[year] = val

        ttm.at[idx] = prev_ttm

    return ttm


def add_ttm_columns(dart_df: pd.DataFrame) -> pd.DataFrame:
    """revenue/op_profit/net_income/tax_expense에 대해 _ttm 컬럼 추가.

    호출 전 report_date 오름차순 정렬돼 있어야 한다.
    """
    df = dart_df.sort_values("report_date").reset_index(drop=True)
    for col in ("revenue", "op_profit", "net_income", "tax_expense"):
        df[f"{col}_ttm"] = compute_ttm_series(df, col)
    return df


# ── ROIC 계산 ────────────────────────────────────────────────────────────────

def compute_roic(row: pd.Series | dict) -> float | None:
    """DART parquet 행에서 ROIC 계산.

    op_profit/tax_expense는 TTM(최근 12개월) 기준 — 유량(flow) 항목이라
    분기별 스케일 왜곡을 피하기 위해 add_ttm_columns()로 미리 계산된
    _ttm 컬럼을 사용한다. equity/total_debt/cash는 재무상태표(대차대조표)
    항목이라 특정 시점의 잔액 그 자체이므로 TTM 개념이 필요 없다.
    """
    op_profit   = _safe_float(row, "op_profit_ttm")
    tax_expense = _safe_float(row, "tax_expense_ttm") or 0.0
    equity      = _safe_float(row, "total_equity")
    total_debt  = _safe_float(row, "total_debt") or 0.0
    cash        = _safe_float(row, "cash") or 0.0

    if op_profit is None or equity is None:
        return None

    tax_rate    = tax_expense / (op_profit + 1e-9) if op_profit != 0 else 0.0
    nopat       = op_profit * (1.0 - tax_rate)
    inv_capital = equity + total_debt - cash
    if inv_capital <= 0:
        return None
    return nopat / inv_capital


# ── 5년 CAGR ─────────────────────────────────────────────────────────────────

def compute_cagr(
    history: pd.DataFrame,
    col: str,
    years: int = 5,
    already_sorted: bool = False,
) -> float | None:
    """DART history에서 col의 N년 CAGR 계산.

    already_sorted=True이면 report_date 오름차순 정렬이 이미 되어 있다고 가정하여 정렬 생략.
    """
    if col not in history.columns:
        return None
    df = (history if already_sorted else history.sort_values("report_date")).dropna(subset=[col])
    n_quarters = years * 4
    if len(df) < n_quarters + 1:
        return None
    v_start = _safe_float(df.iloc[-(n_quarters + 1)], col)
    v_end   = _safe_float(df.iloc[-1], col)
    if v_start is None or v_end is None or v_start <= 0 or v_end <= 0:
        return None
    return (v_end / v_start) ** (1.0 / years) - 1.0


# ── Agent A 원본값 추출 ────────────────────────────────────────────────────────

def extract_raw_factors_a(
    fin: pd.Series,
    price: float | None,
    shares: int | None,
    dart_history: pd.DataFrame,
    dart_history_sorted: pd.DataFrame | None = None,
) -> dict[str, float | None]:
    """Agent A 팩터 원본값 추출.

    fin = dart_cache 가장 최근 행 (publish_date 기준).
    dart_history_sorted = report_date 오름차순 정렬된 동일 DataFrame (CAGR 중복 정렬 방지).
    """
    mktcap  = (price * shares) if (price and shares) else None
    equity  = _safe_float(fin, "total_equity")
    revenue = _safe_float(fin, "revenue")
    op_pr   = _safe_float(fin, "op_profit")
    net_inc = _safe_float(fin, "net_income")
    net_inc_ttm = _safe_float(fin, "net_income_ttm")
    assets  = _safe_float(fin, "total_assets")
    liab    = _safe_float(fin, "total_liabilities")
    cur_a   = _safe_float(fin, "current_assets")
    cur_l   = _safe_float(fin, "current_liabilities")
    cfo     = _safe_float(fin, "cfo")
    cash_v  = _safe_float(fin, "cash")
    int_exp = _safe_float(fin, "interest_expense")

    # EPS, PER, PBR — eps/per는 TTM(최근 12개월) 순이익 기준.
    # net_income 원본값을 그대로 쓰면 가장 최근 공시가 연간보고서인 회사만
    # eps가 3~4배 부풀려져 섹터 내 절대수준 비교(percentile/cross_z)가 왜곡된다.
    eps = _safe_div(net_inc_ttm, shares)
    bps = _safe_div(equity, shares)
    per = _safe_div(price, eps) if (eps and eps > 0) else None
    pbr = _safe_div(price, bps) if (bps and bps > 0) else None

    # 5년 성장률 (정렬된 버전 재사용으로 중복 정렬 방지)
    h = dart_history_sorted if dart_history_sorted is not None else dart_history
    sorted_flag = dart_history_sorted is not None
    if "net_income_ttm" in h.columns and "revenue" in h.columns:
        eps_frame = h[["report_date"]].copy()
        eps_frame["_eps_val"] = h["net_income_ttm"] / shares if shares else np.nan
        eps_growth_5y = compute_cagr(eps_frame, "_eps_val", 5, already_sorted=sorted_flag)
        eq_cagr       = compute_cagr(h, "total_equity", 5, already_sorted=sorted_flag)
    else:
        eps_growth_5y = eq_cagr = None

    return {
        "roe":               _safe_div(net_inc, equity),
        "roa":               _safe_div(net_inc, assets),
        "roic":              compute_roic(fin),
        "net_margin":        _safe_div(net_inc, revenue),
        "op_margin":         _safe_div(op_pr, revenue),
        "roe_5y_avg":        None,  # 5년 평균 ROE — build_factor_dataset에서 history 기반 계산
        "eps_per_share":     eps,
        "per":               per,
        "pbr":               pbr,
        "dividend_yield":    None,  # 별도 배당 데이터 수집 필요
        "asset_turnover":    _safe_div(revenue, assets),
        "debt_ratio":        _safe_div(liab, assets),
        "interest_coverage": _safe_div(op_pr, int_exp) if (int_exp and int_exp > 0) else None,
        "current_ratio":     _safe_div(cur_a, cur_l),
        "fcf_margin":        _safe_div(cfo, revenue),
        "fcf_yield":         _safe_div(cfo, mktcap) if mktcap else None,
        "cash_ratio":        _safe_div(cash_v, cur_l),
        "revenue_growth":    None,  # YoY — build_factor_dataset에서 prev_year와 비교
        "eps_growth_5y":     eps_growth_5y,
        "equity_growth_5y":  eq_cagr,
    }


# ── Agent B 원본값 추출 ────────────────────────────────────────────────────────

def extract_raw_factors_b(
    fin: pd.Series,
    fin_prev_year: pd.Series | None,
    price: float | None,
    shares: int | None,
    dart_history: pd.DataFrame,
    prices_df: pd.DataFrame | None,
    ref_date: str | pd.Timestamp,
    dart_history_sorted: pd.DataFrame | None = None,
) -> dict[str, float | None]:
    """Agent B 팩터 원본값 추출.

    dart_history_sorted = report_date 오름차순 정렬된 동일 DataFrame (CAGR 중복 정렬 방지).
    prices_df           = Date-인덱스 DataFrame 또는 Date-컬럼 DataFrame (양쪽 처리).
    """
    mktcap  = (price * shares) if (price and shares) else None
    revenue = _safe_float(fin, "revenue")
    op_pr   = _safe_float(fin, "op_profit")
    net_inc = _safe_float(fin, "net_income")
    revenue_ttm = _safe_float(fin, "revenue_ttm")
    net_inc_ttm = _safe_float(fin, "net_income_ttm")
    gross   = _safe_float(fin, "gross_profit")
    rd      = _safe_float(fin, "rd_expense")
    cash_v  = _safe_float(fin, "cash")

    rev_prev  = _safe_float(fin_prev_year, "revenue") if fin_prev_year is not None else None
    op_prev   = _safe_float(fin_prev_year, "op_profit") if fin_prev_year is not None else None
    ni_prev   = _safe_float(fin_prev_year, "net_income") if fin_prev_year is not None else None

    revenue_growth   = _safe_div(revenue - rev_prev, abs(rev_prev)) if (revenue is not None and rev_prev and rev_prev != 0) else None
    op_profit_growth = _safe_div(op_pr - op_prev, abs(op_prev)) if (op_pr is not None and op_prev and op_prev != 0) else None

    eps_curr = _safe_div(net_inc, shares)
    eps_prev = _safe_div(ni_prev, shares)
    eps_growth = _safe_div(eps_curr - eps_prev, abs(eps_prev)) if (eps_curr is not None and eps_prev and eps_prev != 0) else None

    # CAGR (정렬된 버전 재사용으로 중복 정렬 방지)
    # revenue_ttm/net_income_ttm 기준 — 원본 분기값을 그대로 비교하면 두 비교
    # 시점의 "분기 유형"(단독분기 vs 연간총액)이 어긋날 때 왜곡될 수 있다.
    # TTM은 매 시점 항상 "최근 12개월"이라 비교 시점이 달라도 척도가 일정하다.
    h = dart_history_sorted if dart_history_sorted is not None else dart_history
    sorted_flag = dart_history_sorted is not None
    rev_cagr = compute_cagr(h, "revenue_ttm", 5, already_sorted=sorted_flag)
    if "net_income_ttm" in h.columns:
        eps_frame = h[["report_date"]].copy()
        eps_frame["_eps_col"] = h["net_income_ttm"] / shares if shares else np.nan
        eps_cagr = compute_cagr(eps_frame, "_eps_col", 5, already_sorted=sorted_flag)
    else:
        eps_cagr = None

    # 운영 레버리지: 영업이익증가율 / 매출증가율
    # 매출증가율이 0에 가까우면 나눗셈 결과가 극단값으로 튀므로 생략(None) 처리
    op_leverage = (
        _safe_div(op_profit_growth, revenue_growth)
        if revenue_growth and abs(revenue_growth) >= _OP_LEVERAGE_MIN_DENOM
        else None
    )

    # PEG, PSR — TTM(최근 12개월) 순이익·매출 기준.
    # 원본값을 그대로 쓰면 가장 최근 공시가 연간보고서인 회사만 eps/매출이
    # 3~4배 부풀려져 psr이 비정상적으로 저평가돼 보이는 문제가 있었다.
    eps_val = _safe_div(net_inc_ttm, shares)
    per_val = _safe_div(price, eps_val) if (eps_val and eps_val > 0) else None
    psr_val = _safe_div(price, _safe_div(revenue_ttm, shares)) if revenue_ttm and shares else None
    peg_val = _safe_div(per_val, eps_growth * 100) if (per_val and eps_growth and eps_growth > 0) else None

    # 가격 기반 팩터 (Date-인덱스 또는 컬럼 방식 자동 처리)
    mom_3m = mom_6m = high52w = None
    if prices_df is not None and not prices_df.empty:
        ref = pd.Timestamp(ref_date)
        # Date-인덱스면 그대로, 컬럼 방식이면 변환 (최소 오버헤드)
        if "Date" in prices_df.columns:
            df_p = prices_df.set_index(pd.to_datetime(prices_df["Date"])).sort_index()
        else:
            df_p = prices_df   # 이미 Date-인덱스
        cur_price_row = df_p[df_p.index <= ref]
        if not cur_price_row.empty:
            p_now = cur_price_row.iloc[-1]["Close"]
            p_3m = df_p[df_p.index <= ref - pd.Timedelta(days=90)]
            p_6m = df_p[df_p.index <= ref - pd.Timedelta(days=180)]
            if not p_3m.empty:
                mom_3m = (p_now - p_3m.iloc[-1]["Close"]) / p_3m.iloc[-1]["Close"]
            if not p_6m.empty:
                mom_6m = (p_now - p_6m.iloc[-1]["Close"]) / p_6m.iloc[-1]["Close"]
            p_1y = df_p[df_p.index >= ref - pd.Timedelta(days=365)]
            if not p_1y.empty:
                hi = p_1y["High"].max()
                lo = p_1y["Low"].min()
                high52w = (p_now - lo) / (hi - lo) if hi != lo else None

    return {
        "roic":               compute_roic(fin),
        "revenue_growth":     revenue_growth,
        "op_profit_growth":   op_profit_growth,
        "eps_growth":         eps_growth,
        "revenue_cagr_5y":    rev_cagr,
        "eps_cagr_5y":        eps_cagr,
        "gross_margin_trend": _safe_div(gross, revenue),
        "operating_leverage": op_leverage,
        "rd_ratio":           _safe_div(rd, revenue),
        "employee_growth":    None,  # 종업원 수 데이터 별도 수집 필요
        "peg":                peg_val,
        "psr":                psr_val,
        "cash_to_mktcap":     _safe_div(cash_v, mktcap) if mktcap else None,
        "momentum_3m":        mom_3m,
        "momentum_6m":        mom_6m,
        "high52w_pct":        high52w,
    }


# ── 이진 팩터 ─────────────────────────────────────────────────────────────────

def compute_binary_a(
    dart_history: pd.DataFrame,
    cfo_current: float | None,
    equity_current: float | None,
    return_3m: float | None,
    return_3m_bottom10pct: float,
) -> dict[str, float]:
    """Agent A 이진 팩터."""
    # 최근 5년(20분기) 연속 흑자
    streak_5y = 0.0
    if "net_income" in dart_history.columns:
        recent = dart_history.sort_values("report_date").tail(20)
        if len(recent) >= 20:
            vals = recent["net_income"].dropna()
            if len(vals) >= 20 and (vals > 0).all():
                streak_5y = 1.0

    return {
        "fcf_positive":     1.0 if (cfo_current is not None and cfo_current > 0) else 0.0,
        "profit_streak_5y": streak_5y,
        "equity_positive":  1.0 if (equity_current is not None and equity_current > 0) else 0.0,
        "no_value_trap":    1.0 if (return_3m is not None and return_3m > return_3m_bottom10pct) else 0.0,
    }


def compute_binary_b(
    dart_history: pd.DataFrame,
    equity_current: float | None,
) -> dict[str, float]:
    """Agent B 이진 팩터."""
    streak_2y = 0.0
    if "net_income" in dart_history.columns:
        recent = dart_history.sort_values("report_date").tail(8)
        if len(recent) >= 8:
            vals = recent["net_income"].dropna()
            if len(vals) >= 8 and (vals > 0).all():
                streak_2y = 1.0

    return {
        "profit_streak_2y": streak_2y,
        "equity_positive":  1.0 if (equity_current is not None and equity_current > 0) else 0.0,
    }


# ── 시계열 팩터 ───────────────────────────────────────────────────────────────

def compute_sector_momentum(
    sector_tickers: list[str],
    price_loader,
    ref_date: str | pd.Timestamp,
    mktcap_map: dict[str, float],
    kospi200_return_1m: float,
) -> float | None:
    """
    섹터 모멘텀 = 시총 가중 1개월 수익률 - KOSPI200 1개월 수익률.
    단순 평균 금지.
    """
    ref     = pd.Timestamp(ref_date)
    start1m = ref - pd.Timedelta(days=30)

    weighted, weights = 0.0, 0.0
    for t in sector_tickers:
        df = price_loader(t)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        now_row = df[df.index <= ref]
        past_row = df[df.index >= start1m].head(1)
        if now_row.empty or past_row.empty:
            continue
        r = (now_row.iloc[-1]["Close"] - past_row.iloc[0]["Close"]) / past_row.iloc[0]["Close"]
        w = mktcap_map.get(t, 0.0)
        weighted += r * w
        weights  += w

    if weights == 0:
        return None
    return weighted / weights - kospi200_return_1m


def compute_sector_momentum_fast(
    sector_tickers: list[str],
    price_mem: dict,          # Date-인덱스 DataFrame (copy·변환 없이 직접 사용)
    ref_date: pd.Timestamp,
    mktcap_map: dict[str, float],
    kospi200_return_1m: float,
) -> float | None:
    """
    compute_sector_momentum의 고속 버전.
    price_mem은 이미 Date-인덱스로 로드된 dict → copy/set_index/sort 생략.
    """
    start1m = ref_date - pd.Timedelta(days=30)

    weighted, weights = 0.0, 0.0
    for t in sector_tickers:
        df = price_mem.get(t)
        if df is None or df.empty:
            continue
        now_row  = df[df.index <= ref_date]
        past_row = df[(df.index >= start1m) & (df.index <= ref_date)].head(1)
        if now_row.empty or past_row.empty:
            continue
        p_now  = now_row.iloc[-1]["Close"]
        p_past = past_row.iloc[0]["Close"]
        if p_past == 0:
            continue
        r = (p_now - p_past) / p_past
        w = mktcap_map.get(t, 0.0)
        weighted += r * w
        weights  += w

    if weights == 0:
        return None
    return weighted / weights - kospi200_return_1m


def compute_ear_3d(
    ticker: str,
    publish_date: pd.Timestamp,
    price_loader,
) -> tuple[float | None, float | None]:
    """
    ear_3d_final, ear_3d_trend 계산.

    day0 = publish_date 직전 영업일 종가 (공시 전 베이스라인)
    day1 = publish_date 종가
    day2 = publish_date + 1영업일 종가
    day3 = publish_date + 2영업일 종가 (3일 대기 완료)
    """
    from utils.calendar_utils import add_trading_days

    df = price_loader(ticker)
    if df is None or df.empty:
        return None, None

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    def _close(dt: pd.Timestamp) -> float | None:
        past = df[df.index <= dt]
        return float(past.iloc[-1]["Close"]) if not past.empty else None

    try:
        c0 = _close(add_trading_days(publish_date, -1))
        c1 = _close(publish_date)
        c2 = _close(add_trading_days(publish_date, 1))
        c3 = _close(add_trading_days(publish_date, 2))
    except Exception:
        return None, None

    if c0 is None or c0 == 0 or c3 is None:
        return None, None

    ear_final = (c3 - c0) / c0
    ear_trend = None
    if c1 is not None and c2 is not None:
        avg12 = (c1 + c2) / 2.0
        if avg12 != 0:
            ear_trend = c3 / avg12 - 1.0

    return ear_final, ear_trend


def compute_ear_3d_fast(
    ticker: str,
    publish_date: pd.Timestamp,
    price_mem: dict,          # Date-인덱스 DataFrame
) -> tuple[float | None, float | None]:
    """
    compute_ear_3d의 고속 버전.
    price_mem은 이미 Date-인덱스로 로드된 dict → copy/set_index/sort 생략.
    """
    from utils.calendar_utils import add_trading_days

    df = price_mem.get(ticker)
    if df is None or df.empty:
        return None, None

    def _close(dt: pd.Timestamp) -> float | None:
        past = df[df.index <= dt]
        return float(past.iloc[-1]["Close"]) if not past.empty else None

    try:
        c0 = _close(add_trading_days(publish_date, -1))
        c1 = _close(publish_date)
        c2 = _close(add_trading_days(publish_date, 1))
        c3 = _close(add_trading_days(publish_date, 2))
    except Exception:
        return None, None

    if c0 is None or c0 == 0 or c3 is None:
        return None, None

    ear_final = (c3 - c0) / c0
    ear_trend = None
    if c1 is not None and c2 is not None:
        avg12 = (c1 + c2) / 2.0
        if avg12 != 0:
            ear_trend = c3 / avg12 - 1.0

    return ear_final, ear_trend


# ── 고수준 인터페이스 ──────────────────────────────────────────────────────────

def apply_feature_transform(
    raw_values: dict[str, float | None],
    factor_meta: list[tuple[str, bool, str]],
    sector_pools: dict[str, np.ndarray],
    all_pools:    dict[str, np.ndarray],
    history_pools: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    raw_values의 각 팩터를 factor_meta 기준으로 3개 피처로 변환.

    Parameters
    ----------
    sector_pools  : {factor_name: 섹터 내 모든 종목 현재값 배열}
    all_pools     : {factor_name: 전체 KOSPI200 현재값 배열}
    history_pools : {factor_name: 해당 종목 과거 8분기 배열}
    """
    features: dict[str, float] = {}
    for factor_name, higher_is_better, scope in factor_meta:
        value = raw_values.get(factor_name)
        feats = generate_continuous_features(
            factor_name,
            value,
            higher_is_better,
            sector_pools.get(factor_name, np.array([])),
            all_pools.get(factor_name, np.array([])),
            history_pools.get(factor_name, np.array([])),
            scope,
        )
        features.update(feats)
    return features
