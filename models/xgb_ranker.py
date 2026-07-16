"""
XGBoost Ranker 학습 파이프라인 (Agent A / B 공통).

설계 원칙:
  - objective = rank:ndcg (소규모 섹터 풀에서 rank:pairwise보다 안정적)
  - group 파라미터 = 매주 금요일 날짜별 종목 수 배열
  - fit 전 반드시 date 오름차순 정렬
  - 원본값 피처 제외, 추가 정규화 없음
  - WFV 확장 윈도우: ab_train_start=2015 고정, fold별 끝점 이동

실행: python -m models.xgb_ranker [A|B] [fold_test_year]
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import ParameterGrid

from config.settings import (
    AGENT_A_FEATURES,
    AGENT_B_FEATURES,
    BACKTEST_DIR,
    MODELS_DIR,
    WFV_CONFIG,
    XGB_PARAM_SEARCH,
)
from utils.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = BACKTEST_DIR / "results"
SAVED_DIR   = MODELS_DIR / "saved"
ARCHIVE_DIR = MODELS_DIR / "archive"


def _load_dataset(agent: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"factor_dataset_{agent}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"factor_dataset_{agent}.parquet 없음. "
            "python -m data.build_factor_dataset 를 먼저 실행하세요."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _feature_cols(agent: str) -> list[str]:
    return AGENT_A_FEATURES if agent == "A" else AGENT_B_FEATURES


def _make_group(df: pd.DataFrame) -> np.ndarray:
    """날짜별 종목 수 배열 — XGBoost fit()의 group 파라미터."""
    return df.groupby("date")["ticker"].count().values


def _to_relevance_grades(df: pd.DataFrame) -> np.ndarray:
    """
    label_3m(연속형 초과수익률)을 rank:ndcg용 사분위 정수 관련도(0~3)로 변환.
    날짜별(쿼리별) 독립 변환 — 날짜 간 수익률 비교 불가 문제 방지.
    """
    # Series로 생성해야 df.index 기반 라벨 할당이 정확히 작동
    grades = pd.Series(0.0, index=df.index, dtype=np.float32)
    for _, grp in df.groupby("date"):
        if len(grp) < 2:
            grades[grp.index] = 1.0  # 단독 종목은 중간 등급
            continue
        q = pd.qcut(grp["label_3m"], q=min(4, len(grp)),
                    labels=False, duplicates="drop")
        grades[grp.index] = q.fillna(0).astype(np.float32)
    return grades.values


def train_fold(
    agent: str,
    df: pd.DataFrame,
    train_end_year: int,
    test_year: int,
    params: dict,
    validate: bool = True,
) -> tuple[xgb.XGBRanker, dict]:
    """
    단일 WFV fold 학습.

    Parameters
    ----------
    train_end_year : 학습 데이터 마지막 연도 (포함)
    test_year      : 테스트 연도
    validate       : True면 테스트셋 NDCG 평가 포함

    Returns
    -------
    (fitted_model, metrics_dict)
    """
    feature_cols = _feature_cols(agent)
    ab_start = WFV_CONFIG["ab_train_start"]

    # label_3m이 있는 행만 사용
    df_valid = df[df["label_3m"].notna()].copy()
    df_valid = df_valid.sort_values("date").reset_index(drop=True)

    # WFV fold 경계 라벨 누수 방지: label_cutoff = test_end - 65 영업일
    from config.settings import LABEL_CONFIG
    from utils.calendar_utils import add_trading_days

    test_end = pd.Timestamp(year=test_year, month=12, day=31)
    label_cutoff = add_trading_days(test_end, -LABEL_CONFIG["label_cutoff_days"])

    train_mask = (
        (df_valid["date"].dt.year >= ab_start) &
        (df_valid["date"].dt.year <= train_end_year) &
        (df_valid["date"] <= label_cutoff)
    )
    test_mask = (df_valid["date"].dt.year == test_year)

    df_train = df_valid[train_mask].copy()
    df_test  = df_valid[test_mask].copy()

    if df_train.empty:
        raise ValueError(f"학습 데이터 없음: {ab_start}~{train_end_year}")
    if df_test.empty:
        logger.warning("테스트 데이터 없음: %d", test_year)

    X_train = df_train[feature_cols].values.astype(np.float32)
    # rank:ndcg는 비음정수 관련도 점수 필요 → 날짜별 사분위 등급(0~3) 변환
    y_train = _to_relevance_grades(df_train)
    g_train = _make_group(df_train)

    model_params = {k: v for k, v in params.items()
                    if k not in ("early_stopping_rounds",)}
    model_params["objective"]   = "rank:ndcg"
    model_params["tree_method"] = "hist"
    model_params["eval_metric"] = "ndcg"

    early = params.get("early_stopping_rounds", 50)

    # XGBoost >= 2.0: early_stopping_rounds는 생성자 파라미터
    if validate:
        model_params["early_stopping_rounds"] = early

    model = xgb.XGBRanker(**model_params)

    if validate and not df_test.empty:
        X_test    = df_test[feature_cols].values.astype(np.float32)
        y_test_gr = _to_relevance_grades(df_test)           # eval_set용 정수 등급
        y_test_ic = df_test["label_3m"].values.astype(np.float32)  # IC 계산용 원본 수익률
        g_test    = _make_group(df_test)
        model.fit(
            X_train, y_train,
            group=g_train,
            eval_set=[(X_test, y_test_gr)],
            eval_group=[g_test],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, group=g_train, verbose=False)

    metrics = {
        "agent":           agent,
        "train_start":     ab_start,
        "train_end":       train_end_year,
        "test_year":       test_year,
        "n_train":         len(df_train),
        "n_test":          len(df_test),
        "best_iteration":  getattr(model, "best_iteration", None),
    }

    if validate and not df_test.empty:
        scores = model.predict(X_test)
        # Spearman IC (테스트셋)
        from scipy.stats import spearmanr
        ic = spearmanr(scores, y_test_ic).correlation  # 원본 수익률 기준 IC
        metrics["ic_test"] = round(float(ic), 4)
        logger.info(
            "Fold (Agent %s) %d→%d test=%d | IC=%.4f | n_train=%d",
            agent, ab_start, train_end_year, test_year, ic, len(df_train),
        )

    return model, metrics


def hyperparameter_search(
    agent: str,
    df: pd.DataFrame,
    fold1_test_year: int = 2019,
) -> dict:
    """
    fold 1 (ab_train_start ~ fold1_test_year-1 → fold1_test_year)을
    기준으로 XGB_PARAM_SEARCH 범위에서 최적 하이퍼파라미터 탐색.

    Returns
    -------
    best_params dict
    """
    logger.info("Agent %s 하이퍼파라미터 탐색 시작 (fold 1 기준)", agent)
    search_grid = {k: v for k, v in XGB_PARAM_SEARCH.items()
                   if isinstance(v, list)}

    best_ic = -np.inf
    best_params: dict = {}

    for params in ParameterGrid(search_grid):
        params["early_stopping_rounds"] = XGB_PARAM_SEARCH.get("early_stopping_rounds", 50)
        try:
            _, metrics = train_fold(
                agent, df,
                train_end_year=fold1_test_year - 1,
                test_year=fold1_test_year,
                params=params,
                validate=True,
            )
            ic = metrics.get("ic_test", -np.inf)
            if ic > best_ic:
                best_ic = ic
                best_params = params.copy()
        except Exception as e:
            logger.debug("파라미터 실패: %s — %s", params, e)

    logger.info("최적 파라미터 (IC=%.4f): %s", best_ic, best_params)
    return best_params


def run_wfv(agent: str, params: dict | None = None) -> list[dict]:
    """
    WFV 5-fold 전체 실행.

    Parameters
    ----------
    params : None이면 먼저 hyperparameter_search() 실행

    Returns
    -------
    fold별 metrics 리스트

    체크포인트: backtest/results/wfv_checkpoint_{agent}.json
    fold 완료마다 즉시 저장 → 중단 후 재시작 가능
    """
    df = _load_dataset(agent)
    feature_cols = _feature_cols(agent)

    # 피처 컬럼 일관성 검증
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise KeyError(f"factor_dataset_{agent}에 피처 없음: {missing}")

    if params is None:
        params = hyperparameter_search(agent, df, fold1_test_year=WFV_CONFIG["test_years"][0])

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 체크포인트 로드
    ckpt_path = RESULTS_DIR / f"wfv_checkpoint_{agent}.json"
    if ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as f:
            ckpt = json.load(f)
        all_metrics: list[dict] = ckpt.get("metrics", [])
        completed:   set[str]   = set(ckpt.get("completed_folds", []))
        logger.info("Agent %s 체크포인트 로드: %d fold 완료됨", agent, len(completed))
    else:
        all_metrics = []
        completed   = set()

    for i, test_year in enumerate(WFV_CONFIG["test_years"]):
        fold_key  = str(test_year)
        fold_path = SAVED_DIR / f"xgb_agent_{agent}_fold{i+1}.pkl"

        # 체크포인트: 완료 fold 스킵
        if fold_key in completed:
            logger.info("Fold %d (%d) — 체크포인트 스킵", i + 1, test_year)
            continue

        train_end = test_year - 1
        logger.info("=== Fold %d: Agent %s %d~%d → %d ===",
                    i + 1, agent, WFV_CONFIG["ab_train_start"], train_end, test_year)

        model, metrics = train_fold(agent, df, train_end, test_year, params, validate=True)
        all_metrics.append(metrics)

        # fold 모델 저장
        with open(fold_path, "wb") as f:
            pickle.dump(model, f)

        # 체크포인트 저장 (fold 완료 즉시)
        completed.add(fold_key)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(
                {"completed_folds": list(completed), "metrics": all_metrics},
                f, ensure_ascii=False, indent=2,
            )

    # 최종 결과 저장
    results_path = RESULTS_DIR / f"wfv_metrics_{agent}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    pass_count = sum(1 for m in all_metrics if m.get("ic_test", 0) > 0)
    logger.info(
        "Agent %s WFV 완료: %d/%d fold IC > 0",
        agent, pass_count, len(all_metrics),
    )
    return all_metrics


def train_full(agent: str, params: dict) -> xgb.XGBRanker:
    """
    전기간(ab_train_start ~ 현재) 재학습 — 실운용 모델.
    WFV pass_threshold 통과 확인 후에만 호출.
    """
    df = _load_dataset(agent)
    df_valid = df[df["label_3m"].notna()].copy()
    df_valid = df_valid.sort_values("date").reset_index(drop=True)

    feature_cols = _feature_cols(agent)
    X = df_valid[feature_cols].values.astype(np.float32)
    # rank:ndcg는 정수 관련도 등급 필요 (fold 학습과 동일 방식 — 원본 연속값 사용 시 XGBoostError)
    y = _to_relevance_grades(df_valid)
    g = _make_group(df_valid)

    model_params = {k: v for k, v in params.items()
                    if k not in ("early_stopping_rounds",)}
    model_params["objective"]   = "rank:ndcg"
    model_params["tree_method"] = "hist"

    model = xgb.XGBRanker(**model_params)
    model.fit(X, y, group=g, verbose=False)

    # 저장 전 기존 모델 백업
    out_path = SAVED_DIR / f"xgb_agent_{agent}.pkl"
    if out_path.exists():
        import shutil
        from datetime import datetime
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.today().strftime("%Y%m%d")
        shutil.copy(out_path, ARCHIVE_DIR / f"xgb_agent_{agent}_{stamp}.pkl")

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Agent %s 전기간 모델 저장: %s", agent, out_path)
    return model


def load_model(agent: str) -> xgb.XGBRanker | None:
    """
    저장된 실운용 모델 로드.
    실제 파일명은 "_live" 접미사가 붙어있다(agent_c.py가 이미 이 이름으로
    로드해 실거래에 쓰고 있음). 접미사 없는 이름은 존재한 적이 없어
    이전에는 이 함수가 항상 None을 반환하고 있었다.
    """
    path = SAVED_DIR / f"xgb_agent_{agent}_live.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_scores(
    model: xgb.XGBRanker,
    df: pd.DataFrame,
    agent: str,
) -> np.ndarray:
    """
    주어진 DataFrame에 대해 XGBoost 점수 계산.

    학습(train_fold/train_full)은 결측치를 raw NaN 그대로 XGBoost에 넘겨
    자체 missing-value 분기 로직으로 처리하게 한다 — 그래서 예측도 동일하게
    raw NaN을 넘겨야 학습 때 배운 분기 방향이 그대로 재현된다.
    (예전엔 fillna(0)을 썼는데, 이러면 "결측"이 아니라 "진짜 0"이라는 값으로
    오인되어 학습 시 결측치에 대해 배운 처리와 다른 경로로 갈라져 실제 종목
    선정이 달라지는 문제가 있었다 — 2026-07-09 발견.)
    """
    feature_cols = _feature_cols(agent)
    X = df[feature_cols].values.astype(np.float32)
    return model.predict(X)


def get_feature_importance(agent: str) -> pd.DataFrame:
    """저장된 모델의 feature importance 반환."""
    model = load_model(agent)
    if model is None:
        raise FileNotFoundError(f"xgb_agent_{agent}.pkl 없음")
    feature_cols = _feature_cols(agent)
    imp = model.feature_importances_
    return pd.DataFrame({
        "feature":    feature_cols,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    agent_arg = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
    assert agent_arg in ("A", "B")
    metrics = run_wfv(agent_arg)
    print(f"\nAgent {agent_arg} WFV 결과:")
    for m in metrics:
        print(f"  Fold {m['test_year']}: IC={m.get('ic_test', 'N/A')}")
