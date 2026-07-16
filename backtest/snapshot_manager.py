"""
MetaML 학습 데이터 주간 스냅샷 저장 및 label_3m 사후 채움.

저장 경로: backtest/results/weekly_snapshots/{YYYY}/{YYYYMMDD}.parquet
{YYYY}는 스냅샷 저장 날짜 기준 연도 (fold 연도 아님).

Step 4-B: 스냅샷 저장 (금요일 15:30 실행)
Step 4-C: label_3m 사후 채움 (T+65 이후 — 금요일 루틴에 포함)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import BACKTEST_DIR, LABEL_CONFIG
from utils.calendar_utils import add_trading_days, count_trading_days
from utils.logger import get_logger

logger = get_logger(__name__)

SNAPSHOT_BASE = BACKTEST_DIR / "results" / "weekly_snapshots"


def save_weekly_snapshot(
    ref_date: str | pd.Timestamp,
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    signal_c: pd.DataFrame,
    portfolio_tickers: list[str],
    macro_features: dict | None = None,
) -> Path:
    """
    금요일 15:30 기준 주간 포트폴리오 스냅샷 저장.

    Parameters
    ----------
    ref_date          : 스냅샷 기준일 (금요일)
    scores_a          : Agent A XGBoost 점수 DataFrame (ticker, xgb_score)
    scores_b          : Agent B XGBoost 점수 DataFrame (ticker, xgb_score)
    signal_c          : Agent C 신호 DataFrame (ticker, cnn_score, lstm_score, final_score)
    portfolio_tickers : 현재 보유 종목 목록
    macro_features    : 거시 지표 딕셔너리 (선택)

    Returns
    -------
    저장된 parquet 경로
    """
    ref = pd.Timestamp(ref_date)
    year_dir = SNAPSHOT_BASE / ref.strftime("%Y")
    year_dir.mkdir(parents=True, exist_ok=True)
    out_path = year_dir / f"{ref.strftime('%Y%m%d')}.parquet"

    # A/B 점수 병합
    scores_a = scores_a.rename(columns={"xgb_score": "score_a"})
    scores_b = scores_b.rename(columns={"xgb_score": "score_b"})
    signal_c = signal_c.rename(columns={"final_score": "signal_c"})

    merged = scores_a.merge(scores_b, on="ticker", how="outer", suffixes=("_a", "_b"))
    merged = merged.merge(signal_c[["ticker", "cnn_score", "lstm_score", "signal_c"]],
                          on="ticker", how="outer")

    n = len(merged)
    merged["date"]             = ref.strftime("%Y-%m-%d")
    merged["ab_hold_ratio"]    = merged["ticker"].apply(
        lambda t: 1.0 if t in portfolio_tickers else 0.0
    )
    merged["c_hold_days"]      = 0.0  # 실운용 연동 시 채워짐
    merged["signal_consensus"] = _compute_consensus(merged)
    merged["a_c_product"]      = merged.get("score_a", pd.Series([np.nan]*n)) * merged.get("signal_c", pd.Series([np.nan]*n))
    merged["b_c_product"]      = merged.get("score_b", pd.Series([np.nan]*n)) * merged.get("signal_c", pd.Series([np.nan]*n))
    merged["a_b_diff"]         = merged.get("score_a", pd.Series([np.nan]*n)).fillna(0) - merged.get("score_b", pd.Series([np.nan]*n)).fillna(0)

    # 거시 피처 (있으면 broadcast)
    if macro_features:
        for k, v in macro_features.items():
            merged[k] = v
    else:
        for col in ["vix_z", "hy_spread", "dxy_z", "usdkrw_z"]:
            merged[col] = np.nan

    # 사후 채움 컬럼 (나중에 fill_snapshot_labels()에서 업데이트)
    merged["alpha_3m"]       = None
    merged["c_actual_return"] = None
    merged["composite_label"] = None

    merged.to_parquet(out_path, index=False)
    logger.info("MetaML 스냅샷 저장: %s (%d 종목)", out_path, n)
    return out_path


def _compute_consensus(df: pd.DataFrame) -> pd.Series:
    """A·B·C 신호 일치도 (모두 양수 or 모두 음수 → 높음)."""
    def _consensus(row):
        a = row.get("score_a", np.nan)
        b = row.get("score_b", np.nan)
        c = row.get("signal_c", np.nan)
        vals = [v for v in [a, b, c] if not (v is None or (isinstance(v, float) and np.isnan(v)))]
        if len(vals) < 2:
            return np.nan
        signs = [1 if v > 0 else -1 for v in vals]
        return sum(signs) / len(signs)  # 1.0=전부동의, -1.0=전부반대
    return df.apply(_consensus, axis=1)


def fill_snapshot_labels(ref_date_str: str | None = None) -> None:
    """
    Step 4-C: T+65 이후 경과한 스냅샷의 alpha_3m, composite_label 채움.
    ref_date_str=None이면 오늘 기준으로 전체 스캔.
    """
    from data.build_price_cache import load_price_cache

    today = pd.Timestamp.today()
    cutoff_bd = LABEL_CONFIG["label_cutoff_days"]

    for year_dir in sorted(SNAPSHOT_BASE.glob("*")):
        if not year_dir.is_dir():
            continue
        for parquet_path in sorted(year_dir.glob("*.parquet")):
            snap_date = pd.Timestamp(parquet_path.stem)
            if count_trading_days(snap_date, today) < cutoff_bd:
                continue  # 아직 T+65 미경과

            df = pd.read_parquet(parquet_path)
            if "alpha_3m" not in df.columns:
                df["alpha_3m"] = None
            if "composite_label" not in df.columns:
                df["composite_label"] = None

            needs_fill = df["alpha_3m"].isna()
            if not needs_fill.any():
                continue

            updated = 0
            for idx in df[needs_fill].index:
                ticker = df.at[idx, "ticker"]
                price_df = load_price_cache(ticker)
                if price_df is None:
                    continue
                price_df["Date"] = pd.to_datetime(price_df["Date"])
                price_df = price_df.set_index("Date").sort_index()

                t1 = add_trading_days(snap_date, 1)
                t1_row = price_df[price_df.index >= t1].head(1)
                if t1_row.empty:
                    continue
                entry_open = float(t1_row.iloc[0]["Open"])
                if entry_open == 0:
                    continue

                avg_prices = []
                for offset in range(61, 66):
                    dt = add_trading_days(snap_date, offset)
                    row_p = price_df[price_df.index <= dt]
                    if not row_p.empty:
                        avg_prices.append(float(row_p.iloc[-1]["Close"]))
                if len(avg_prices) < 3:
                    continue

                # KOSPI200 수익률 (T+1 시가 → T+63 종가 구간)
                from data.build_factor_dataset import _get_kospi200_period_return
                kospi_ret = _get_kospi200_period_return(t1, add_trading_days(snap_date, 63))

                alpha_3m = np.mean(avg_prices) / entry_open - 1 - kospi_ret
                df.at[idx, "alpha_3m"] = round(alpha_3m, 6)

                c_ret = df.at[idx, "c_actual_return"]
                if c_ret is not None and not (isinstance(c_ret, float) and np.isnan(c_ret)):
                    df.at[idx, "composite_label"] = round(0.5 * alpha_3m + 0.5 * float(c_ret), 6)

                updated += 1

            if updated > 0:
                df.to_parquet(parquet_path, index=False)
                logger.info("스냅샷 라벨 채움: %s (%d행)", parquet_path.name, updated)


def list_snapshots() -> pd.DataFrame:
    """저장된 스냅샷 목록 반환."""
    rows = []
    for year_dir in sorted(SNAPSHOT_BASE.glob("*")):
        for p in sorted(year_dir.glob("*.parquet")):
            df = pd.read_parquet(p)
            rows.append({
                "date":          p.stem,
                "n_tickers":     len(df),
                "label_filled":  df["alpha_3m"].notna().sum() if "alpha_3m" in df.columns else 0,
            })
    return pd.DataFrame(rows)
