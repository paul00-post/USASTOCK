"""
백테스팅 엔진 v2 — 실제 주수·현금 기반 (확정 사양 v2).

핵심 변경사항 (v1 → v2):
  - 비율(weight) 기반 → 주수(share)/현금(원화) 기반
  - D일 시가 기준 일괄 처리: 갭 필터 → 예산 계산 → 부분매도 → 매수
  - 목표 비중 11% (portfolio_value × 0.11), 허용 범위 +3%
  - 단일 종목 건너뜀: 1주 가격 > portfolio_value × 40%
  - 단일 종목 1회 상한: portfolio_value × 15% (1주 예외 허용)
  - 현금 부족 시: 최소 TP거리 부분매도 → 최대 보유일 부분매도 → forced_cash_shortage
  - max_sector_position: 30% → 50%

공통 원칙:
  - 백테스팅 = 실운용 파이프라인과 완전 동일 (CNN+LSTM 100%)
  - CNN/LSTM 미학습 시 signal=0.0 폴백 (거래 비활성)
  - A 20개 + B 20개 분리 선별 (통합 방식 금지)
  - is_holding = ticker in new_portfolio (강제청산 이후 상태 기준)
  - 청산 우선순위: 익절 → 손절 → 4일 연속 sell → 타이머
  - ta.index < date (당일 미포함 룩어헤드 방지)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    INITIAL_CAPITAL,
    RISK_LIMITS,
    SIGNAL_THRESHOLDS,
    TRADE_CONFIG,
    TRANSACTION_COSTS,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# 거래비용 헬퍼 — 미국 주식 수수료 체계(2026-07-16 한국투자증권 공식 페이지 확인).
# 국내와 달리 별도 거래세(tax)가 없고, 대신 매도 시에만 SEC Fee + FINRA TAF가
# 붙는다. FINRA TAF는 정률이 아니라 "1주당 고정 금액"이라 buy_cost처럼 가격에
# 곱하는 방식이 아니라 1주당 정액으로 차감한다.
_COMM      = TRANSACTION_COSTS["commission"]
_SEC_FEE   = TRANSACTION_COSTS["sec_fee"]
_TAF       = TRANSACTION_COSTS["finra_taf_per_share"]
_SLIP      = TRANSACTION_COSTS["slippage"]


def buy_cost(price: float) -> float:
    return price * (1 + _COMM + _SLIP)


def sell_proceeds(price: float) -> float:
    """1주당 순매도 대금 (수수료+SEC Fee+슬리피지 정률 차감 + FINRA TAF 정액 차감).

    FINRA TAF의 건당 상한($9.79)은 이 프로젝트의 계좌 규모(포지션당 수십~
    수백 주)에서는 사실상 발동하지 않는 수준(5만주 이상 필요)이라 반영하지
    않는다 — 계좌 규모가 훨씬 커지면 재검토 필요.
    """
    return price * (1 - _COMM - _SEC_FEE - _SLIP) - _TAF


@dataclass
class Position:
    ticker:           str
    shares:           int
    entry_price:      float
    entry_date:       pd.Timestamp
    take_profit_price:float
    stop_loss_price:  float
    hold_days:        int = 0
    sell_signal_days: int = 0   # 연속 sell 신호 카운터
    halted:           bool = False  # 거래정지 여부


@dataclass
class Portfolio:
    cash:         float = INITIAL_CAPITAL
    holdings:     dict[str, Position] = field(default_factory=dict)
    pending_buys: dict[str, float]   = field(default_factory=dict)  # ticker → signal score
    trade_log:    list[dict] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)

    def total_value(self, prices: dict[str, float]) -> float:
        stock_value = sum(
            pos.shares * prices.get(pos.ticker, pos.entry_price)
            for pos in self.holdings.values()
        )
        return self.cash + stock_value

    def portfolio_weights(self, prices: dict[str, float]) -> dict[str, float]:
        total = self.total_value(prices)
        if total == 0:
            return {}
        return {
            t: pos.shares * prices.get(t, pos.entry_price) / total
            for t, pos in self.holdings.items()
        }


def _compute_atr(price_df: pd.DataFrame, ref_date: pd.Timestamp, window: int = 10) -> float | None:
    """ATR 계산. ref_date 당일 미포함 (< 기준)."""
    past = price_df[price_df["Date"] < ref_date].tail(window + 1)
    if len(past) < 2:
        return None
    high  = past["High"].values
    low   = past["Low"].values
    close = past["Close"].values
    tr    = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:]  - close[:-1]),
        ),
    )
    return float(tr.mean()) if len(tr) > 0 else None


def _get_ohlc(price_df: pd.DataFrame, date: pd.Timestamp) -> tuple | None:
    """해당 날짜의 OHLC 반환. 없으면 None."""
    row = price_df[price_df["Date"] == date]
    if row.empty:
        return None
    r = row.iloc[0]
    return float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])


def calc_stop_loss(
    price_df: pd.DataFrame,
    ref_date: pd.Timestamp,
    n: int = 10,
    buffer: float = 0.01,
) -> float | None:
    """과거 N영업일 Low 최솟값 × (1 - buffer). ref_date 당일 미포함."""
    past = price_df[price_df["Date"] < ref_date].tail(n)
    if len(past) < 1:
        return None
    return float(past["Low"].min()) * (1 - buffer)


def calc_take_profit(entry_price: float, stop_loss: float, rr: float = 2.0) -> float:
    """entry + (entry - SL) × R:R"""
    return entry_price + (entry_price - stop_loss) * rr


def _calc_position_budget(portfolio_val: float) -> tuple[float, float]:
    """
    목표 비중 11%, 허용 +3% 기준 매수 예산 범위.

    Returns
    -------
    (budget_low, budget_high)  — 원화 기준
    """
    low  = portfolio_val * RISK_LIMITS["target_position_pct"]
    high = low + portfolio_val * RISK_LIMITS["budget_range_pct"]
    return low, high


def _partial_sell_for_cash(
    portfolio: "Portfolio",
    exclude_ticker: str,
    cash_needed: float,
    open_price_map: dict[str, float],
    date: pd.Timestamp,
    strategy: str,  # "min_tp_distance" | "max_hold_days"
) -> None:
    """
    현금 확보를 위해 기존 보유 종목 일부 매도.

    strategy:
      "min_tp_distance" — TP까지 남은 거리 최소 종목 (청산 임박 포지션)
      "max_hold_days"   — 보유 기간 최장 종목
    """
    candidates = {
        t: pos for t, pos in portfolio.holdings.items()
        if t != exclude_ticker and pos.shares > 0 and not pos.halted
        and t in open_price_map
    }
    if not candidates:
        return

    if strategy == "min_tp_distance":
        # TP까지 남은 거리 = take_profit_price - current_open
        target = min(
            candidates.items(),
            key=lambda kv: kv[1].take_profit_price - open_price_map[kv[0]]
        )
    else:  # max_hold_days
        target = max(candidates.items(), key=lambda kv: kv[1].hold_days)

    t_sell, pos_sell = target
    cur_open = open_price_map[t_sell]
    net_per_share = sell_proceeds(cur_open)
    if net_per_share <= 0:
        return

    additional_needed = max(0.0, cash_needed - portfolio.cash)
    shares_to_sell = min(
        int(np.ceil(additional_needed / net_per_share)),
        pos_sell.shares,
    )
    if shares_to_sell <= 0:
        return

    proceeds = net_per_share * shares_to_sell
    portfolio.cash += proceeds
    pos_sell.shares -= shares_to_sell

    portfolio.trade_log.append({
        "ticker":       t_sell,
        "entry_date":   pos_sell.entry_date.strftime("%Y-%m-%d"),
        "exit_date":    date.strftime("%Y-%m-%d"),
        "entry_price":  pos_sell.entry_price,
        "exit_price":   cur_open,
        "shares":       shares_to_sell,
        "return_pct":   round((cur_open - pos_sell.entry_price) / pos_sell.entry_price, 4),
        "hold_days":    pos_sell.hold_days,
        "exit_reason":  f"forced_cash_shortage_{strategy}",
    })

    # 잔여 주수 0 → 포지션 제거
    if pos_sell.shares <= 0:
        portfolio.holdings.pop(t_sell, None)


def _sector_cap(n_in_universe: int) -> int:
    """섹터 내 유니버스 종목 수 → 최대 선정 가능 수."""
    if n_in_universe <= 4:
        return 1
    elif n_in_universe <= 10:
        return 2
    return 999  # 제한 없음


def _select_with_sector_cap(
    scores: pd.DataFrame,
    ticker_to_sector: dict[str, str],
    sector_universe_counts: dict[str, int],
    n: int = 40,
) -> list[str]:
    """
    xgb_score 내림차순 정렬 후 섹터 상한 규칙을 적용하여 최대 n개 선정.
    섹터 미분류 종목은 포함 가능 (상한 적용 제외).
    """
    sorted_scores = scores.sort_values("xgb_score", ascending=False)
    selected: list[str] = []
    sector_selected: dict[str, int] = {}

    for ticker in sorted_scores["ticker"]:
        if len(selected) >= n:
            break
        sec = ticker_to_sector.get(ticker)
        if sec is None:
            selected.append(ticker)
            continue
        cap = _sector_cap(sector_universe_counts.get(sec, 0))
        if sector_selected.get(sec, 0) >= cap:
            continue
        selected.append(ticker)
        sector_selected[sec] = sector_selected.get(sec, 0) + 1

    return selected


def run_backtest(
    agent_a_scores: pd.DataFrame,   # columns: date, ticker, xgb_score
    agent_b_scores: pd.DataFrame,
    cnn_signals: pd.DataFrame | None,   # columns: date, ticker, cnn_score, lstm_score, final_score
    lstm_signals: pd.DataFrame | None,
    price_loader,                        # callable(ticker) → pd.DataFrame | None
    start_date: str | pd.Timestamp,
    end_date:   str | pd.Timestamp,
    alpha: float = 0.5,
    pool_mode: str = "AB",              # "AB"=A+B합산, "A"=A만, "B"=B만
    buy_threshold: float | None = None,        # None → SIGNAL_THRESHOLDS["buy_upper"] 사용
    super_buy_threshold: float | None = None,  # None → 단일 임계값 (two-tier 없음)
    pool_n: int | None = None,                 # None → 기본값 (A/B 단독:40, AB:20)
    buy_position_pct: float | None = None,       # None → RISK_LIMITS["target_position_pct"] 사용
    super_buy_position_pct: float | None = None, # None → RISK_LIMITS["max_single_position"] 사용
    no_partial_sell: bool = False,               # True → 현금 부족 시 매수 스킵 (부분매도 없음)
) -> tuple[Portfolio, pd.DataFrame]:
    """
    전략 백테스팅 실행.

    Parameters
    ----------
    agent_a_scores : A 에이전트 XGBoost 점수 (일별)
    agent_b_scores : B 에이전트 XGBoost 점수
    cnn_signals    : CNN 신호 (None이면 0.0 폴백)
    lstm_signals   : LSTM 신호 (None이면 0.0 폴백)
    price_loader   : ticker → OHLCV DataFrame
    alpha          : CNN 가중치 (β = 1-α 자동)

    Returns
    -------
    (Portfolio, equity_curve_df)
    """
    from utils.calendar_utils import get_trading_days

    portfolio = Portfolio()
    _buy_thr = buy_threshold if buy_threshold is not None else SIGNAL_THRESHOLDS["buy_upper"]
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    trading_days = get_trading_days(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    # 섹터 맵 (리스크 한도 계산용 + pool 섹터 상한)
    from data.sector_classifier import load_sector_map
    sector_map = load_sector_map()
    # ticker → custom_sector 빠른 조회용 딕셔너리 (pool 섹터 상한에서 반복 사용)
    _ticker_to_sector: dict[str, str] = (
        sector_map.dropna(subset=["custom_sector"])
        .set_index("ticker")["custom_sector"]
        .to_dict()
    )

    # ── 버그①: A/B 점수 forward-fill ──────────────────────────────────────────
    # factor_dataset은 금요일 스냅샷만 존재 → 월~목은 직전 금요일 점수 사용
    # sort + ffill로 각 날짜에 "가장 최근 금요일 점수"를 매핑
    def _ffill_scores(df_scores: pd.DataFrame) -> pd.DataFrame:
        if df_scores.empty:
            return df_scores
        df_scores = df_scores.sort_values("date")
        all_dates = pd.DataFrame({"date": trading_days})
        tickers_in = df_scores["ticker"].unique()
        filled_parts = []
        for t in tickers_in:
            sub = df_scores[df_scores["ticker"] == t][["date", "ticker", "xgb_score"]]
            sub = all_dates.merge(sub, on="date", how="left")
            sub["ticker"]    = t
            sub["xgb_score"] = sub["xgb_score"].ffill()
            filled_parts.append(sub.dropna(subset=["xgb_score"]))
        return pd.concat(filled_parts, ignore_index=True) if filled_parts else df_scores

    agent_a_scores = _ffill_scores(agent_a_scores)
    agent_b_scores = _ffill_scores(agent_b_scores)
    # ─────────────────────────────────────────────────────────────────────────

    # ── 가격 데이터 사전 로드 (루프 내 반복 disk I/O 방지) ──────────────────
    # pool 후보 전체를 백테스트 시작 전 한 번만 로드해 메모리에 캐싱
    _all_pool_tickers: set[str] = set(agent_b_scores["ticker"].tolist())
    if not agent_a_scores.empty:
        _all_pool_tickers |= set(agent_a_scores["ticker"].tolist())
    _price_cache_global: dict[str, pd.DataFrame] = {}
    for _t in _all_pool_tickers:
        _df = price_loader(_t)
        if _df is not None:
            _df["Date"] = pd.to_datetime(_df["Date"])
            _price_cache_global[_t] = _df
    # ────────────────────────────────────────────────────────────────────────

    for date in trading_days:
        # ── Pool 결정 (pool_mode: "AB"=A+B합산, "A"=A만, "B"=B만) ──
        # 섹터 상한 규칙: 유니버스 내 섹터 종목 수 ≤4→1개, 5~10→2개, >10→제한없음
        # AB: A/B 각각 섹터 상한 적용 후 최대 20개 → 합산 최대 40
        # 단독(A or B): 섹터 상한 적용 후 최대 40개
        top_a: list[str] = []
        top_b: list[str] = []
        n_single = pool_n if pool_n is not None else (40 if pool_mode in ("A", "B") else 20)

        if pool_mode in ("AB", "A"):
            a_today = agent_a_scores[agent_a_scores["date"] == date]
            if not a_today.empty:
                a_universe = a_today["ticker"].tolist()
                a_sec_counts = {}
                for t in a_universe:
                    s = _ticker_to_sector.get(t)
                    if s:
                        a_sec_counts[s] = a_sec_counts.get(s, 0) + 1
                top_a = _select_with_sector_cap(a_today, _ticker_to_sector, a_sec_counts, n_single)

        if pool_mode in ("AB", "B"):
            b_today = agent_b_scores[agent_b_scores["date"] == date]
            if not b_today.empty:
                b_universe = b_today["ticker"].tolist()
                b_sec_counts = {}
                for t in b_universe:
                    s = _ticker_to_sector.get(t)
                    if s:
                        b_sec_counts[s] = b_sec_counts.get(s, 0) + 1
                top_b = _select_with_sector_cap(b_today, _ticker_to_sector, b_sec_counts, n_single)

        pool = list(dict.fromkeys(top_a + top_b))  # 중복 제거, 순서 보존

        # ── 가격 수집 (캐시 조회 — 루프 밖에서 사전 로드됨) ────────────────────
        current_prices: dict[str, float] = {}
        for t in set(pool) | set(portfolio.holdings.keys()) | set(portfolio.pending_buys.keys()):
            if t not in _price_cache_global:
                # 예상치 못한 신규 종목: 처음 1회만 로드 후 캐시에 저장
                _df2 = price_loader(t)
                if _df2 is not None:
                    _df2["Date"] = pd.to_datetime(_df2["Date"])
                    _price_cache_global[t] = _df2
            df = _price_cache_global.get(t)
            if df is not None:
                ohlc = _get_ohlc(df, date)
                if ohlc:
                    current_prices[t] = ohlc[3]  # Close
        price_cache = _price_cache_global  # 하위 코드 호환성

        # ── 보유 종목 거래정지 상태 갱신 ──
        for t, pos in portfolio.holdings.items():
            if t not in current_prices:
                pos.halted = True
            else:
                if pos.halted:
                    pos.halted = False

        # ── 청산 처리 (보유 종목 루프) ──
        to_exit: list[tuple[str, str, float]] = []  # (ticker, reason, price)

        for t, pos in list(portfolio.holdings.items()):
            if pos.halted:
                pos.hold_days += 1
                # 5일 초과 시 알림
                if pos.hold_days % TRADE_CONFIG.get("halt_alert_days", 5) == 0:
                    logger.warning("거래정지 %d영업일 초과: %s — 관리자 확인 필요", pos.hold_days, t)
                continue

            ohlc = _get_ohlc(price_cache.get(t, pd.DataFrame()), date)
            if ohlc is None:
                pos.hold_days += 1
                continue
            _, high, low, close = ohlc

            # Step 1: 익절
            if high >= pos.take_profit_price:
                to_exit.append((t, "take_profit", pos.take_profit_price))
                continue

            # Step 2: 손절
            if low <= pos.stop_loss_price:
                to_exit.append((t, "stop_loss", pos.stop_loss_price))
                continue

            # Step 3: 4일 연속 sell 신호
            signal = _get_final_score(t, date, cnn_signals, lstm_signals, alpha)
            if signal < 0:
                pos.sell_signal_days += 1
            else:
                pos.sell_signal_days = 0

            if pos.sell_signal_days >= SIGNAL_THRESHOLDS["sell_consecutive"]:
                to_exit.append((t, "signal_exit", close))
                pos.sell_signal_days = 0
                continue

            # Step 4: 타이머 청산
            pos.hold_days += 1
            if pos.hold_days >= TRADE_CONFIG["max_hold_days"]:
                to_exit.append((t, "max_hold_exit", close))
                continue

        # 청산 실행
        new_portfolio = dict(portfolio.holdings)
        for t, reason, exit_price in to_exit:
            if t not in new_portfolio:
                continue
            pos = new_portfolio.pop(t)
            proceeds = sell_proceeds(exit_price) * pos.shares
            portfolio.cash += proceeds
            ret_pct = (exit_price - pos.entry_price) / pos.entry_price
            portfolio.trade_log.append({
                "ticker":       t,
                "entry_date":   pos.entry_date.strftime("%Y-%m-%d"),
                "exit_date":    date.strftime("%Y-%m-%d"),
                "entry_price":  pos.entry_price,
                "exit_price":   exit_price,
                "shares":       pos.shares,
                "return_pct":   round(ret_pct, 4),
                "hold_days":    pos.hold_days,
                "exit_reason":  reason,
            })
        portfolio.holdings = new_portfolio

        # ── 신규 매수: 전일 pending → 오늘 시가 체결 (v2: 주수·현금 기반) ────────
        _gap_mult = TRADE_CONFIG["gap_filter_mult"]
        _gap_win  = TRADE_CONFIG["gap_atr_window"]
        _sl_win   = TRADE_CONFIG["swing_low_window"]
        _sl_buf   = TRADE_CONFIG["swing_low_buffer"]
        _tp_rr    = TRADE_CONFIG["take_profit_rr"]

        # D일 시가 맵 (부분매도 헬퍼에 전달)
        dday_open_map: dict[str, float] = {}
        for _t, _df in price_cache.items():
            _o = _get_ohlc(_df, date)
            if _o:
                dday_open_map[_t] = _o[0]

        executed_or_skipped: list[str] = []
        for t, prev_score in portfolio.pending_buys.items():
            if t in portfolio.holdings:
                executed_or_skipped.append(t)
                continue

            df_p = price_cache.get(t)
            if df_p is None:
                executed_or_skipped.append(t)
                continue

            ohlc = _get_ohlc(df_p, date)
            if ohlc is None:
                executed_or_skipped.append(t)
                continue
            open_price, _, _, _ = ohlc

            # 전일 종가 (갭 필터용)
            prev_rows = df_p[df_p["Date"] < date]
            if prev_rows.empty:
                executed_or_skipped.append(t)
                continue
            prev_close = float(prev_rows.iloc[-1]["Close"])

            # ATR(10) 갭 필터
            atr10 = _compute_atr(df_p, date, window=_gap_win)
            if atr10 is None:
                executed_or_skipped.append(t)
                continue
            gap = open_price - prev_close
            if gap > _gap_mult * atr10:
                executed_or_skipped.append(t)
                logger.debug("%s 갭 필터 스킵: gap=%.0f > ATR×%.1f=%.0f", t, gap, _gap_mult, _gap_mult * atr10)
                continue

            # 스윙 저점 손절가
            sl = calc_stop_loss(df_p, date, n=_sl_win, buffer=_sl_buf)
            if sl is None:
                executed_or_skipped.append(t)
                continue

            entry_price = open_price

            # 역전 방지: SL >= entry → 진입 취소
            if sl >= entry_price:
                logger.debug("%s 진입 취소: SL(%.0f) >= entry(%.0f)", t, sl, entry_price)
                executed_or_skipped.append(t)
                continue

            tp = calc_take_profit(entry_price, sl, _tp_rr)

            # ── v2: 주수·현금 기반 포지션 사이징 ────────────────────────────────
            portfolio_val    = portfolio.total_value(current_prices)
            one_share_cost   = buy_cost(entry_price)
            skip_threshold   = portfolio_val * RISK_LIMITS["skip_single_pct"]   # 40%

            # 1주 가격 > 40% → 건너뜀
            if one_share_cost > skip_threshold:
                executed_or_skipped.append(t)
                logger.debug("%s 건너뜀: 1주비용(%.0f) > 40%%(%.0f)", t, one_share_cost, skip_threshold)
                continue

            # 예산 결정 (two-tier: buy vs super_buy 비중 각각 적용)
            _buy_pct  = buy_position_pct if buy_position_pct is not None else RISK_LIMITS["target_position_pct"]
            _sbuy_pct = super_buy_position_pct if super_buy_position_pct is not None else RISK_LIMITS["max_single_position"]

            if super_buy_threshold is not None and prev_score >= super_buy_threshold:
                effective_budget = portfolio_val * _sbuy_pct
            else:
                effective_budget = portfolio_val * _buy_pct

            # 주수 결정
            shares = int(effective_budget / one_share_cost)
            if shares == 0:
                # 1주 예외: 2-tier 모드(super_buy_threshold 설정)에서는 강한 신호(≥super_buy)만 허용
                # 단일 임계값 모드(super_buy_threshold=None)에서는 기존처럼 항상 허용
                is_strong_or_single_tier = (
                    super_buy_threshold is None or prev_score >= super_buy_threshold
                )
                if is_strong_or_single_tier:
                    shares = 1
                else:
                    # 약한 신호(buy tier, 0.05~0.075): 예산 초과 시 스킵 (1주 예외 없음)
                    executed_or_skipped.append(t)
                    logger.debug(
                        "%s 약신호 예산 초과 스킵: signal=%.3f budget=%.0f cost=%.0f",
                        t, prev_score, effective_budget, one_share_cost,
                    )
                    continue

            # 섹터 비중 체크 (50% 상한)
            sector_row = sector_map[sector_map["ticker"] == t]
            sector = sector_row.iloc[0]["custom_sector"] if not sector_row.empty else None
            if sector and portfolio_val > 0:
                sec_value = sum(
                    portfolio.holdings[tt].shares * current_prices.get(tt, 0)
                    for tt in portfolio.holdings
                    if not sector_map[sector_map["ticker"] == tt].empty
                    and sector_map[sector_map["ticker"] == tt]["custom_sector"].values[0] == sector
                )
                new_value = one_share_cost * shares
                if (sec_value + new_value) / portfolio_val > RISK_LIMITS["max_sector_position"]:
                    executed_or_skipped.append(t)
                    logger.debug("%s 섹터 한도 초과: %s", t, sector)
                    continue

            # 현금 부족 시 처리
            cash_needed = one_share_cost * shares
            if portfolio.cash < cash_needed:
                if no_partial_sell:
                    # 부분매도 없이 즉시 스킵
                    executed_or_skipped.append(t)
                    logger.debug(
                        "%s 현금부족 스킵(no_partial_sell): 필요=%.0f, 현금=%.0f",
                        t, cash_needed, portfolio.cash
                    )
                    continue
                # 부분매도로 현금 확보 (우선순위: min TP거리 → max 보유일)
                _partial_sell_for_cash(
                    portfolio, t, cash_needed, dday_open_map, date, "min_tp_distance"
                )
            if not no_partial_sell and portfolio.cash < cash_needed:
                _partial_sell_for_cash(
                    portfolio, t, cash_needed, dday_open_map, date, "max_hold_days"
                )
            if portfolio.cash < cash_needed:
                executed_or_skipped.append(t)
                logger.debug(
                    "%s forced_cash_shortage: 필요=%.0f, 현금=%.0f",
                    t, cash_needed, portfolio.cash
                )
                continue

            portfolio.cash -= cash_needed
            portfolio.holdings[t] = Position(
                ticker=t,
                shares=shares,
                entry_price=entry_price,
                entry_date=date,
                take_profit_price=tp,
                stop_loss_price=sl,
            )
            executed_or_skipped.append(t)
            logger.debug(
                "매수 체결: %s %d주 @%.0f원 (SL=%.0f, TP=%.0f, 예산=%.0f)",
                t, shares, entry_price, sl, tp, effective_budget,
            )

        for t in executed_or_skipped:
            portfolio.pending_buys.pop(t, None)

        # ── 내일 매수 후보 결정 ─────────────────────────────────────────────────
        for t in pool:
            if t in portfolio.holdings or t in portfolio.pending_buys:
                continue
            signal = _get_final_score(t, date, cnn_signals, lstm_signals, alpha)
            if signal >= _buy_thr:
                portfolio.pending_buys[t] = signal

        # ── 자산 평가액 기록 ──
        ev = portfolio.total_value(current_prices)
        portfolio.equity_curve.append((date, ev))

    trade_log_df = pd.DataFrame(portfolio.trade_log)
    equity_df = pd.DataFrame(portfolio.equity_curve, columns=["date", "equity"])
    equity_df = equity_df.set_index("date")

    return portfolio, equity_df


def run_b_only_backtest(
    agent_b_scores: pd.DataFrame,
    price_loader,
    start_date: str | pd.Timestamp,
    end_date:   str | pd.Timestamp,
) -> tuple[Portfolio, pd.DataFrame]:
    """
    구성 4: Agent B 단독 (C 없음).
    매주 금요일 Top20 결정 → 다음 영업일 시가에 이탈 종목 매도·진입 종목 매수.
    유지 종목은 거래 없음 (익절/손절/타이머 없음).
    """
    from utils.calendar_utils import get_trading_days

    portfolio = Portfolio()
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    trading_days = get_trading_days(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    # B 점수 forward-fill (금요일만 있으므로 월~목은 직전 금요일 값 사용)
    def _ffill(df_scores: pd.DataFrame) -> pd.DataFrame:
        if df_scores.empty:
            return df_scores
        df_scores = df_scores.sort_values("date")
        all_dates = pd.DataFrame({"date": trading_days})
        parts = []
        for t in df_scores["ticker"].unique():
            sub = df_scores[df_scores["ticker"] == t][["date", "ticker", "xgb_score"]]
            sub = all_dates.merge(sub, on="date", how="left")
            sub["ticker"]    = t
            sub["xgb_score"] = sub["xgb_score"].ffill()
            parts.append(sub.dropna(subset=["xgb_score"]))
        return pd.concat(parts, ignore_index=True) if parts else df_scores

    agent_b_scores = _ffill(agent_b_scores)

    # pending 리밸런싱: {ticker: "sell" | "buy"}  금요일 결정 → 다음 영업일 체결
    pending_rebalance: dict[str, str] = {}
    first_day = True

    for date in trading_days:
        relevant = set(portfolio.holdings.keys()) | set(pending_rebalance.keys())

        # 가격 수집
        price_cache: dict[str, pd.DataFrame] = {}
        current_prices: dict[str, float] = {}
        for t in relevant:
            df = price_loader(t)
            if df is not None:
                df["Date"] = pd.to_datetime(df["Date"])
                price_cache[t] = df
                ohlc = _get_ohlc(df, date)
                if ohlc:
                    current_prices[t] = ohlc[3]

        # ── 리밸런싱 체결 (초기 진입 또는 전 금요일 결정분) ──────────────────
        if first_day or pending_rebalance:
            if first_day:
                # 최초 진입: 시작일 기준 Top20 전체 매수
                b_today = agent_b_scores[agent_b_scores["date"] == date]
                top20 = b_today.nlargest(20, "xgb_score")["ticker"].tolist() if not b_today.empty else []
                for t in top20:
                    if t not in price_cache:
                        df = price_loader(t)
                        if df is not None:
                            df["Date"] = pd.to_datetime(df["Date"])
                            price_cache[t] = df
                            ohlc = _get_ohlc(df, date)
                            if ohlc:
                                current_prices[t] = ohlc[3]
                total_val   = portfolio.cash
                target_per  = total_val / max(len(top20), 1)
                for t in top20:
                    df_p = price_cache.get(t)
                    if df_p is None:
                        continue
                    ohlc = _get_ohlc(df_p, date)
                    if ohlc is None:
                        continue
                    open_price = ohlc[0]
                    if open_price <= 0:
                        continue
                    shares = int(target_per / buy_cost(open_price))
                    if shares <= 0 or portfolio.cash < buy_cost(open_price) * shares:
                        continue
                    portfolio.cash -= buy_cost(open_price) * shares
                    portfolio.holdings[t] = Position(
                        ticker=t, shares=shares,
                        entry_price=open_price, entry_date=date,
                        take_profit_price=float("inf"), stop_loss_price=0.0,
                    )
                first_day = False
            else:
                # 매도 먼저 (이탈 종목)
                for t, action in list(pending_rebalance.items()):
                    if action != "sell" or t not in portfolio.holdings:
                        continue
                    df_p = price_cache.get(t)
                    if df_p is None:
                        continue
                    ohlc = _get_ohlc(df_p, date)
                    if ohlc is None:
                        continue
                    pos = portfolio.holdings.pop(t)
                    proceeds = sell_proceeds(ohlc[0]) * pos.shares
                    portfolio.cash += proceeds
                    portfolio.trade_log.append({
                        "ticker":      t,
                        "entry_date":  pos.entry_date.strftime("%Y-%m-%d"),
                        "exit_date":   date.strftime("%Y-%m-%d"),
                        "entry_price": pos.entry_price,
                        "exit_price":  ohlc[0],
                        "shares":      pos.shares,
                        "return_pct":  round((ohlc[0] - pos.entry_price) / pos.entry_price, 4),
                        "hold_days":   pos.hold_days,
                        "exit_reason": "rebalance",
                    })

                # 매수 (신규 진입 종목) — 목표 비중 = total / 20
                total_val  = portfolio.total_value(current_prices)
                target_per = total_val / 20
                for t, action in list(pending_rebalance.items()):
                    if action != "buy":
                        continue
                    if t not in price_cache:
                        df = price_loader(t)
                        if df is not None:
                            df["Date"] = pd.to_datetime(df["Date"])
                            price_cache[t] = df
                            ohlc = _get_ohlc(df, date)
                            if ohlc:
                                current_prices[t] = ohlc[3]
                    df_p = price_cache.get(t)
                    if df_p is None:
                        continue
                    ohlc = _get_ohlc(df_p, date)
                    if ohlc is None:
                        continue
                    open_price = ohlc[0]
                    if open_price <= 0:
                        continue
                    shares = int(target_per / buy_cost(open_price))
                    if shares <= 0 or portfolio.cash < buy_cost(open_price) * shares:
                        continue
                    portfolio.cash -= buy_cost(open_price) * shares
                    portfolio.holdings[t] = Position(
                        ticker=t, shares=shares,
                        entry_price=open_price, entry_date=date,
                        take_profit_price=float("inf"), stop_loss_price=0.0,
                    )
            pending_rebalance = {}

        # ── 금요일: 다음 리밸런싱 결정 ──────────────────────────────────────
        if date.dayofweek == 4:
            b_today = agent_b_scores[agent_b_scores["date"] == date]
            new_top20 = set(b_today.nlargest(20, "xgb_score")["ticker"].tolist()) if not b_today.empty else set()
            current_holdings = set(portfolio.holdings.keys())
            for t in current_holdings - new_top20:
                pending_rebalance[t] = "sell"
            for t in new_top20 - current_holdings:
                pending_rebalance[t] = "buy"

        # hold_days 증가
        for pos in portfolio.holdings.values():
            pos.hold_days += 1

        # 자산 평가액 기록
        ev = portfolio.total_value(current_prices)
        portfolio.equity_curve.append((date, ev))

    trade_log_df = pd.DataFrame(portfolio.trade_log)
    equity_df    = pd.DataFrame(portfolio.equity_curve, columns=["date", "equity"])
    equity_df    = equity_df.set_index("date")
    return portfolio, equity_df


def _get_final_score(
    ticker: str,
    date: pd.Timestamp,
    cnn_signals: pd.DataFrame | None,
    lstm_signals: pd.DataFrame | None,
    alpha: float,
) -> float:
    """CNN/LSTM 신호에서 final_score 계산. 미학습 시 0.0 폴백.

    MultiIndex(date, ticker) DataFrame이면 O(log n) .loc[] 조회,
    일반 DataFrame이면 기존 boolean 필터 사용.
    """
    cnn_score  = 0.0
    lstm_score = 0.0

    if cnn_signals is not None:
        if isinstance(cnn_signals.index, pd.MultiIndex):
            try:
                cnn_score = float(cnn_signals.loc[(date, ticker), "cnn_score"])
            except KeyError:
                pass
        else:
            row = cnn_signals[
                (cnn_signals["date"] == date) & (cnn_signals["ticker"] == ticker)
            ]
            if not row.empty:
                cnn_score = float(row.iloc[0].get("cnn_score", 0.0))

    if lstm_signals is not None:
        if isinstance(lstm_signals.index, pd.MultiIndex):
            try:
                lstm_score = float(lstm_signals.loc[(date, ticker), "lstm_score"])
            except KeyError:
                pass
        else:
            row = lstm_signals[
                (lstm_signals["date"] == date) & (lstm_signals["ticker"] == ticker)
            ]
            if not row.empty:
                lstm_score = float(row.iloc[0].get("lstm_score", 0.0))

    return alpha * cnn_score + (1 - alpha) * lstm_score
