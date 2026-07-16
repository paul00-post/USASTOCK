"""
IC 스크리닝 — 1차 노이즈 제거용 필터.

역할: WFV 5-fold 중 2번 이상 |IC| > 0.002 통과한 피처만 생존.
     최종 선별은 XGBoost feature importance에 위임.
     강한 필터로 쓰지 말 것 (단변량 IC가 낮아도 조합 예측력 있을 수 있음).

실행: python -m backtest.ic_screening [A|B]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import (
    AGENT_A_FEATURES,
    AGENT_B_FEATURES,
    BACKTEST_DIR,
    IC_MIN_FOLDS,
    IC_MIN_THRESHOLD,
    MODELS_DIR,
    WFV_CONFIG,
)
from utils.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = BACKTEST_DIR / "results"
SAVED_DIR   = MODELS_DIR / "saved"


def _feature_cols(agent: str) -> list[str]:
    return AGENT_A_FEATURES if agent == "A" else AGENT_B_FEATURES


def compute_ic_by_fold(agent: str) -> dict[str, list[float]]:
    """
    WFV 각 fold의 학습셋 기준 Spearman IC 계산.

    Returns
    -------
    {feature_name: [ic_fold1, ic_fold2, ..., ic_fold5]}
    """
    path = RESULTS_DIR / f"factor_dataset_{agent}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"factor_dataset_{agent}.parquet 없음")

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["label_3m"].notna()].copy()
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = _feature_cols(agent)
    ab_start = WFV_CONFIG["ab_train_start"]

    ic_by_feature: dict[str, list[float]] = {f: [] for f in feature_cols}

    for test_year in WFV_CONFIG["test_years"]:
        train_end = test_year - 1

        from config.settings import LABEL_CONFIG
        from utils.calendar_utils import add_trading_days
        test_end     = pd.Timestamp(year=test_year, month=12, day=31)
        label_cutoff = add_trading_days(test_end, -LABEL_CONFIG["label_cutoff_days"])

        train_df = df[
            (df["date"].dt.year >= ab_start) &
            (df["date"].dt.year <= train_end) &
            (df["date"] <= label_cutoff)
        ]
        if train_df.empty:
            logger.warning("fold %d: 학습 데이터 없음", test_year)
            for f in feature_cols:
                ic_by_feature[f].append(np.nan)
            continue

        y = train_df["label_3m"].values
        for f in feature_cols:
            x = train_df[f].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 20:
                ic_by_feature[f].append(np.nan)
                continue
            ic, _ = spearmanr(x[mask], y[mask])
            ic_by_feature[f].append(float(ic) if not np.isnan(ic) else np.nan)

    return ic_by_feature


def run_screening(agent: str) -> list[str]:
    """
    IC 스크리닝 실행 → 생존 피처 목록 반환 및 JSON 저장.

    기준: |IC| > IC_MIN_THRESHOLD 인 fold 수 >= IC_MIN_FOLDS
    """
    logger.info("Agent %s IC 스크리닝 시작 (|IC|>%.4f, %d-fold 중 %d회 이상)",
                agent, IC_MIN_THRESHOLD, len(WFV_CONFIG["test_years"]), IC_MIN_FOLDS)

    ic_by_feature = compute_ic_by_fold(agent)
    feature_cols  = _feature_cols(agent)

    survived: list[str] = []
    killed:   list[str] = []

    for f in feature_cols:
        ic_vals = ic_by_feature[f]
        pass_count = sum(
            1 for ic in ic_vals
            if ic is not None and not np.isnan(ic) and abs(ic) > IC_MIN_THRESHOLD
        )
        if pass_count >= IC_MIN_FOLDS:
            survived.append(f)
        else:
            killed.append(f)
            logger.debug("제거: %s (통과 %d/%d fold)", f, pass_count, len(ic_vals))

    logger.info("IC 스크리닝 결과: 생존 %d / 제거 %d (전체 %d)",
                len(survived), len(killed), len(feature_cols))

    # 결과 저장
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "agent":        agent,
        "threshold":    IC_MIN_THRESHOLD,
        "min_folds":    IC_MIN_FOLDS,
        "survived":     survived,
        "killed":       killed,
        "ic_by_feature": {
            f: [round(v, 5) if v is not None and not np.isnan(v) else None for v in vs]
            for f, vs in ic_by_feature.items()
        },
    }
    out_path = SAVED_DIR / f"ic_screening_{agent}.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    logger.info("IC 스크리닝 결과 저장: %s", out_path)

    return survived


def load_survived_features(agent: str) -> list[str]:
    """저장된 IC 스크리닝 결과에서 생존 피처 로드."""
    path = SAVED_DIR / f"ic_screening_{agent}.json"
    if not path.exists():
        logger.warning("IC 스크리닝 결과 없음 — 전체 피처 사용")
        return _feature_cols(agent)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["survived"]


def print_ic_summary(agent: str) -> None:
    """IC 결과 요약 출력."""
    path = SAVED_DIR / f"ic_screening_{agent}.json"
    if not path.exists():
        print("IC 스크리닝 결과 없음")
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n=== Agent {agent} IC 스크리닝 결과 ===")
    print(f"생존: {len(data['survived'])} / 제거: {len(data['killed'])} / 전체: {len(data['survived'])+len(data['killed'])}")
    print("\n[생존 피처 IC (fold별)]")
    ic_map = data["ic_by_feature"]
    for f in data["survived"]:
        vals = [f"{v:.4f}" if v is not None else "NaN" for v in ic_map.get(f, [])]
        print(f"  {f:<50} {vals}")
    if data["killed"]:
        print(f"\n[제거 피처 ({len(data['killed'])}개)]: {data['killed'][:10]}{'...' if len(data['killed'])>10 else ''}")


if __name__ == "__main__":
    agent_arg = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
    assert agent_arg in ("A", "B"), "Usage: python -m backtest.ic_screening [A|B]"
    survived = run_screening(agent_arg)
    print_ic_summary(agent_arg)
