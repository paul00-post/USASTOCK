"""
백테스팅 성과 지표 계산.

포함 지표: CAGR, Sharpe(무위험이자율 차감), MDD, 승률, 평균 보유일
Sharpe 계산 시 rf = 연 3.5% 기준금리 / 252 일별로 차감.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RF_ANNUAL = 0.035   # 연 무위험 이자율 3.5%
RF_DAILY  = RF_ANNUAL / 252


def compute_cagr(equity_curve: pd.Series) -> float:
    """
    CAGR 계산.

    Parameters
    ----------
    equity_curve : 날짜 인덱스, 포트폴리오 평가액 시계열
    """
    if len(equity_curve) < 2:
        return 0.0
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return total_return ** (1 / years) - 1


def compute_sharpe(daily_returns: pd.Series) -> float:
    """
    일별 수익률 시계열에서 연환산 Sharpe.
    무위험 이자율 차감 후 계산.
    """
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - RF_DAILY
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def compute_mdd(equity_curve: pd.Series) -> float:
    """최대낙폭(MDD)."""
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return float(drawdown.min())


def compute_hit_rate(trade_log: pd.DataFrame) -> float:
    """거래 로그에서 수익 거래 비율."""
    if trade_log.empty:
        return 0.0
    wins = (trade_log["return_pct"] > 0).sum()
    return wins / len(trade_log)


def compute_exit_breakdown(trade_log: pd.DataFrame) -> dict:
    """exit_reason별 건수 집계."""
    if trade_log.empty or "exit_reason" not in trade_log.columns:
        return {}
    counts = trade_log["exit_reason"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def compute_avg_hold_days(trade_log: pd.DataFrame) -> float:
    """평균 보유 영업일 수."""
    if trade_log.empty:
        return 0.0
    return float(trade_log["hold_days"].mean())


def summarize_performance(
    equity_curve: pd.Series,
    trade_log: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    """
    전체 성과 요약 딕셔너리.

    Parameters
    ----------
    equity_curve       : 날짜 인덱스, 평가액 시계열
    trade_log          : 거래 기록 DataFrame
    benchmark_returns  : KOSPI200 일별 수익률 (None이면 초과수익 계산 생략)
    """
    daily_ret = equity_curve.pct_change().dropna()

    result = {
        "cagr":           round(compute_cagr(equity_curve), 4),
        "sharpe":         round(compute_sharpe(daily_ret), 4),
        "mdd":            round(compute_mdd(equity_curve), 4),
        "hit_rate":       round(compute_hit_rate(trade_log), 4),
        "avg_hold_days":  round(compute_avg_hold_days(trade_log), 1),
        "n_trades":       len(trade_log),
        "exit_breakdown": compute_exit_breakdown(trade_log),
        "total_return":  round(
            (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) >= 2 else 0.0,
            4,
        ),
    }

    if benchmark_returns is not None and len(benchmark_returns) > 1:
        # 벤치마크 초과 수익 (연환산)
        strat_total = (1 + daily_ret).prod() - 1
        bench_total = (1 + benchmark_returns).prod() - 1
        result["alpha_total"] = round(strat_total - bench_total, 4)

        # Information Ratio
        active = daily_ret.values - benchmark_returns.reindex(daily_ret.index).fillna(0).values
        ir_std = np.std(active)
        result["info_ratio"] = round(
            np.mean(active) / ir_std * np.sqrt(252) if ir_std > 0 else 0.0, 4
        )

    return result


def beats_benchmark(metrics: dict, benchmark_cagr: float) -> bool:
    """전략 CAGR이 벤치마크보다 높은지 여부 (WFV pass_threshold 판단용)."""
    return metrics.get("cagr", 0.0) > benchmark_cagr
