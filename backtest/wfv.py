"""
Walk-Forward Validation 실행기.

XGBoost A/B + CNN/LSTM C 전체 WFV 파이프라인.
각 fold 완료마다 checkpoint 저장 (중단 후 재시작 가능).

실행: python -m backtest.wfv [A|B|C|ALL]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    ALPHA_GRID,
    BACKTEST_DIR,
    INITIAL_CAPITAL,
    MODELS_DIR,
    WFV_CONFIG,
)
from backtest.engine import run_backtest
from backtest.metrics import beats_benchmark, summarize_performance
from utils.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR    = BACKTEST_DIR / "results"
CHECKPOINT_PATH = RESULTS_DIR / "wfv_checkpoint.json"
SAVED_DIR      = MODELS_DIR / "saved"


def _load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_folds": [], "results": []}


def _save_checkpoint(checkpoint: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def _get_kospi200_daily_returns(start: str, end: str) -> pd.Series:
    """SPDR S&P 500 ETF(SPY) 일별 수익률 반환 (벤치마크)."""
    from data.build_price_cache import load_price_cache
    df = load_price_cache("SPY")
    if df is None:
        return pd.Series(dtype=float)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[start:end]
    return df["Close"].pct_change().dropna()


def run_ab_wfv(agent: str) -> list[dict]:
    """
    Agent A 또는 B의 XGBoost WFV만 실행 (신호 계산 없이 랭킹 품질만 평가).
    """
    from models.xgb_ranker import hyperparameter_search, run_wfv, _load_dataset

    logger.info("=== Agent %s XGBoost WFV 시작 ===", agent)
    df = _load_dataset(agent)
    params = hyperparameter_search(agent, df, fold1_test_year=WFV_CONFIG["test_years"][0])
    metrics = run_wfv(agent, params=params)
    return metrics


def run_full_wfv(alpha: float = 0.5) -> dict:
    """
    A + B + C 통합 WFV: 각 fold에서 전략 포트폴리오 시뮬레이션.

    반환: {fold_year: performance_metrics}
    """
    from data.build_price_cache import load_price_cache
    from models.xgb_ranker import _load_dataset, train_fold, hyperparameter_search
    from backtest.snapshot_manager import save_weekly_snapshot

    checkpoint   = _load_checkpoint()
    completed    = set(checkpoint.get("completed_folds", []))
    all_results  = checkpoint.get("results", [])
    # fold 1 α 그리드 서치 결과를 체크포인트에서 복원 (fold 1이 스킵될 때를 대비)
    best_alpha   = checkpoint.get("best_alpha", alpha)

    ab_start = WFV_CONFIG["ab_train_start"]

    # 하이퍼파라미터 사전 탐색
    df_a = _load_dataset("A")
    df_b = _load_dataset("B")
    params_a = hyperparameter_search("A", df_a, WFV_CONFIG["test_years"][0])
    params_b = hyperparameter_search("B", df_b, WFV_CONFIG["test_years"][0])

    for i, test_year in enumerate(WFV_CONFIG["test_years"]):
        fold_key = str(test_year)
        if fold_key in completed:
            logger.info("Fold %d (%d) — 체크포인트에서 스킵", i + 1, test_year)
            continue

        train_end = test_year - 1
        logger.info("=== Fold %d: %d~%d 학습 → %d 테스트 ===",
                    i + 1, ab_start, train_end, test_year)

        # A/B XGBoost 학습 + fold 모델 저장 (체크포인트 및 사후 분석용)
        import pickle as _pickle
        model_a, _ = train_fold("A", df_a, train_end, test_year, params_a, validate=False)
        model_b, _ = train_fold("B", df_b, train_end, test_year, params_b, validate=False)
        _fold_a_path = SAVED_DIR / f"xgb_agent_A_fold{i+1}.pkl"
        _fold_b_path = SAVED_DIR / f"xgb_agent_B_fold{i+1}.pkl"
        with open(_fold_a_path, "wb") as _f: _pickle.dump(model_a, _f)
        with open(_fold_b_path, "wb") as _f: _pickle.dump(model_b, _f)

        # 테스트 기간 A/B 점수 생성
        test_start = pd.Timestamp(year=test_year, month=1, day=1)
        test_end   = pd.Timestamp(year=test_year, month=12, day=31)
        from config.settings import AGENT_A_FEATURES, AGENT_B_FEATURES

        def _make_scores(df: pd.DataFrame, model, feat_cols: list[str], agent: str) -> pd.DataFrame:
            # 추론(inference)에는 label_3m 불필요 — 테스트 연도만 필터
            # XGBoost는 NaN 피처를 자체 처리하므로 dropna 불필요
            mask = df["date"].dt.year == test_year
            sub = df[mask].copy()
            if sub.empty:
                return pd.DataFrame(columns=["date", "ticker", "xgb_score"])
            X = sub[feat_cols].values.astype(np.float32)
            sub["xgb_score"] = model.predict(X)
            return sub[["date", "ticker", "xgb_score"]]

        scores_a = _make_scores(df_a, model_a, AGENT_A_FEATURES, "A")
        scores_b = _make_scores(df_b, model_b, AGENT_B_FEATURES, "B")

        # CNN/LSTM 신호 로드 (train_c.py run_c_wfv() 결과물)
        sig_path = RESULTS_DIR / f"signals_c_fold{i+1}.parquet"
        if sig_path.exists():
            sig_df = pd.read_parquet(sig_path)
            sig_df["date"] = pd.to_datetime(sig_df["date"])
            cnn_sigs  = sig_df  # cnn_score 컬럼 포함
            lstm_sigs = sig_df  # lstm_score 컬럼 포함
            logger.info("Fold %d CNN/LSTM 신호 로드: %s (%d행)", i + 1, sig_path.name, len(sig_df))
        else:
            cnn_sigs = lstm_sigs = None
            logger.warning(
                "Fold %d CNN/LSTM 신호 없음 — 0.0 폴백 "
                "(먼저 python -m models.train_c wfv 실행 필요)", i + 1
            )

        # α 그리드 서치 (fold 1에서만 탐색, 이후 fold 1 결과 고정 사용)
        if i == 0:
            best_alpha = _grid_search_alpha(
                scores_a, scores_b, cnn_sigs, lstm_sigs,
                load_price_cache, test_start, test_end,
            )
            logger.info("최적 α = %.1f (fold 1 기준)", best_alpha)
        # else: best_alpha는 fold 1에서 설정된 값 유지 (루프 바깥에서 초기화됨)

        # 포트폴리오 시뮬레이션
        _, equity_df = run_backtest(
            agent_a_scores=scores_a,
            agent_b_scores=scores_b,
            cnn_signals=cnn_sigs,
            lstm_signals=lstm_sigs,
            price_loader=load_price_cache,
            start_date=test_start,
            end_date=test_end,
            alpha=best_alpha,
        )

        # 벤치마크 수익률
        bench_rets = _get_kospi200_daily_returns(
            test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")
        )

        if equity_df.empty:
            metrics = {"test_year": test_year, "error": "equity_curve 비어있음"}
        else:
            from backtest.metrics import summarize_performance
            trade_log_df = pd.DataFrame()  # 상세 기록은 engine에서 별도 관리
            metrics = summarize_performance(equity_df["equity"], trade_log_df, bench_rets)
            metrics["test_year"] = test_year
            metrics["alpha"] = best_alpha
            metrics["beats_benchmark"] = beats_benchmark(
                metrics, float((1 + bench_rets).prod() ** (1 / 1) - 1) if len(bench_rets) > 0 else 0
            )

        all_results.append(metrics)
        completed.add(fold_key)
        # best_alpha도 체크포인트에 저장 — fold 1 스킵 후 재시작 시 복원용
        _save_checkpoint({
            "completed_folds": list(completed),
            "results":         all_results,
            "best_alpha":      best_alpha,
        })

        logger.info(
            "Fold %d 완료: CAGR=%.2f%% Sharpe=%.2f MDD=%.2f%% beats=%s",
            i + 1,
            metrics.get("cagr", 0) * 100,
            metrics.get("sharpe", 0),
            metrics.get("mdd", 0) * 100,
            metrics.get("beats_benchmark", "?"),
        )

    # 최종 결과 저장
    out_path = RESULTS_DIR / "walk_forward_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    pass_count = sum(1 for r in all_results if r.get("beats_benchmark", False))
    logger.info(
        "WFV 전체 완료: %d/%d fold KOSPI200 초과 (threshold=%d)",
        pass_count, len(all_results), WFV_CONFIG["pass_threshold"],
    )
    return {"folds": all_results, "pass_count": pass_count}


def _grid_search_alpha(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    cnn_signals,
    lstm_signals,
    price_loader,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    """fold 1 테스트 기간에서 α ∈ ALPHA_GRID 중 Sharpe 최대 α 반환."""
    from backtest.metrics import compute_sharpe

    best_sharpe = -np.inf
    best_alpha  = 0.5

    for alpha in ALPHA_GRID:
        try:
            _, eq = run_backtest(
                scores_a, scores_b, cnn_signals, lstm_signals,
                price_loader, start_date, end_date, alpha,
            )
            if eq.empty:
                continue
            s = compute_sharpe(eq["equity"].pct_change().dropna())
            if s > best_sharpe:
                best_sharpe = s
                best_alpha  = alpha
        except Exception as e:
            logger.debug("α=%.1f 실패: %s", alpha, e)

    return best_alpha


def print_wfv_summary() -> None:
    """WFV 결과 요약 출력."""
    path = RESULTS_DIR / "walk_forward_results.json"
    if not path.exists():
        print("WFV 결과 없음")
        return
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    print("\n=== WFV 결과 요약 ===")
    print(f"{'연도':<6} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'벤치초과':>8}")
    for r in results:
        print(
            f"{r.get('test_year','?'):<6} "
            f"{r.get('cagr',0)*100:>7.2f}% "
            f"{r.get('sharpe',0):>8.2f} "
            f"{r.get('mdd',0)*100:>7.2f}% "
            f"{'✓' if r.get('beats_benchmark') else '✗':>8}"
        )
    pass_count = sum(1 for r in results if r.get("beats_benchmark", False))
    print(f"\n통과: {pass_count}/{len(results)} fold (기준: {WFV_CONFIG['pass_threshold']})")


# ── 4가지 구성 정의 ──────────────────────────────────────────────────────────
CONFIGS = {
    "AB+C": {"pool_mode": "AB", "use_c": True},   # 구성 1: A+B 풀 + C 신호 (최종 실운용)
    "A+C":  {"pool_mode": "A",  "use_c": True},   # 구성 2: A만 + C 신호
    "B+C":  {"pool_mode": "B",  "use_c": True},   # 구성 3: B만 + C 신호
    "B":    {"pool_mode": "B",  "use_c": False},  # 구성 4: B만, C 없음 (대조군)
}


def run_all_configs(alpha: float = 0.5) -> dict:
    """
    4가지 구성 통합 백테스팅.
    모델 학습(A/B)은 fold당 1회 — 4가지 시뮬레이션에서 공유.
    C 신호는 train_c.py에서 미리 생성된 parquet 로드.

    체크포인트: backtest/results/wfv_checkpoint_all.json
    """
    import pickle as _pickle

    from data.build_price_cache import load_price_cache
    from models.xgb_ranker import _load_dataset, train_fold, hyperparameter_search
    from backtest.engine import run_b_only_backtest
    from config.settings import AGENT_A_FEATURES, AGENT_B_FEATURES

    ckpt_path = RESULTS_DIR / "wfv_checkpoint_all.json"
    if ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as f:
            ckpt = json.load(f)
        logger.info("체크포인트 로드 완료")
    else:
        ckpt = {
            "configs": {name: {"completed_folds": [], "results": []} for name in CONFIGS},
        }

    best_alpha = ckpt.get("best_alpha", alpha)

    df_a = _load_dataset("A")
    df_b = _load_dataset("B")
    params_a = hyperparameter_search("A", df_a, WFV_CONFIG["test_years"][0])
    params_b = hyperparameter_search("B", df_b, WFV_CONFIG["test_years"][0])

    for i, test_year in enumerate(WFV_CONFIG["test_years"]):
        fold_key   = str(test_year)
        train_end  = test_year - 1
        test_start = pd.Timestamp(year=test_year, month=1, day=1)
        test_end   = pd.Timestamp(year=test_year, month=12, day=31)

        remaining = [
            name for name in CONFIGS
            if fold_key not in ckpt["configs"][name]["completed_folds"]
        ]
        if not remaining:
            logger.info("Fold %d (%d) — 전 구성 체크포인트 스킵", i + 1, test_year)
            continue

        logger.info("=== Fold %d: %d~%d 학습 → %d 테스트 (구성: %s) ===",
                    i + 1, WFV_CONFIG["ab_train_start"], train_end, test_year, remaining)

        # A/B 모델: 기존 fold 파일 있으면 재사용
        fold_a_path = SAVED_DIR / f"xgb_agent_A_fold{i+1}.pkl"
        fold_b_path = SAVED_DIR / f"xgb_agent_B_fold{i+1}.pkl"

        if fold_a_path.exists() and fold_b_path.exists():
            with open(fold_a_path, "rb") as f: model_a = _pickle.load(f)
            with open(fold_b_path, "rb") as f: model_b = _pickle.load(f)
            logger.info("Fold %d — A/B fold 모델 로드 (학습 스킵)", i + 1)
        else:
            model_a, _ = train_fold("A", df_a, train_end, test_year, params_a, validate=False)
            model_b, _ = train_fold("B", df_b, train_end, test_year, params_b, validate=False)
            with open(fold_a_path, "wb") as f: _pickle.dump(model_a, f)
            with open(fold_b_path, "wb") as f: _pickle.dump(model_b, f)

        def _make_scores(df: pd.DataFrame, model, feat_cols: list[str]) -> pd.DataFrame:
            sub = df[df["date"].dt.year == test_year].copy()
            if sub.empty:
                return pd.DataFrame(columns=["date", "ticker", "xgb_score"])
            sub["xgb_score"] = model.predict(sub[feat_cols].values.astype(np.float32))
            return sub[["date", "ticker", "xgb_score"]]

        scores_a = _make_scores(df_a, model_a, AGENT_A_FEATURES)
        scores_b = _make_scores(df_b, model_b, AGENT_B_FEATURES)

        sig_path = RESULTS_DIR / f"signals_c_fold{i+1}.parquet"
        if sig_path.exists():
            sig_df = pd.read_parquet(sig_path)
            sig_df["date"] = pd.to_datetime(sig_df["date"])
            cnn_sigs = lstm_sigs = sig_df
        else:
            cnn_sigs = lstm_sigs = None
            logger.warning("Fold %d C 신호 없음 — 0.0 폴백 (먼저 python -m models.train_c wfv 실행)", i + 1)

        bench_rets = _get_kospi200_daily_returns(
            test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")
        )
        bench_annual = float((1 + bench_rets).prod() - 1) if len(bench_rets) > 0 else 0.0

        for config_name in remaining:
            cfg = CONFIGS[config_name]
            logger.info("  [%s] 시뮬레이션 시작", config_name)

            try:
                if cfg["use_c"]:
                    if config_name == "AB+C" and i == 0 and "best_alpha" not in ckpt:
                        best_alpha = _grid_search_alpha(
                            scores_a, scores_b, cnn_sigs, lstm_sigs,
                            load_price_cache, test_start, test_end,
                        )
                        ckpt["best_alpha"] = best_alpha
                        logger.info("  최적 α = %.1f (fold 1 AB+C 기준)", best_alpha)

                    _, equity_df = run_backtest(
                        scores_a, scores_b, cnn_sigs, lstm_sigs,
                        load_price_cache, test_start, test_end,
                        alpha=best_alpha, pool_mode=cfg["pool_mode"],
                    )
                else:
                    _, equity_df = run_b_only_backtest(
                        scores_b, load_price_cache, test_start, test_end,
                    )

                if equity_df.empty:
                    metrics = {"test_year": test_year, "config": config_name,
                               "error": "equity_curve 비어있음"}
                else:
                    metrics = summarize_performance(equity_df["equity"], pd.DataFrame(), bench_rets)
                    metrics["test_year"]       = test_year
                    metrics["config"]          = config_name
                    metrics["alpha"]           = best_alpha if cfg["use_c"] else None
                    metrics["beats_benchmark"] = beats_benchmark(metrics, bench_annual)

            except Exception as e:
                logger.error("  [%s] Fold %d 실패: %s", config_name, i + 1, e)
                metrics = {"test_year": test_year, "config": config_name, "error": str(e)}

            ckpt["configs"][config_name]["results"].append(metrics)
            ckpt["configs"][config_name]["completed_folds"].append(fold_key)
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(ckpt, f, ensure_ascii=False, indent=2)

            logger.info(
                "  [%s] Fold %d 완료: CAGR=%.2f%% Sharpe=%.2f MDD=%.2f%% beats=%s",
                config_name, i + 1,
                metrics.get("cagr", 0) * 100, metrics.get("sharpe", 0),
                metrics.get("mdd", 0) * 100,
                "✓" if metrics.get("beats_benchmark") else "✗",
            )

    # 최종 결과 저장
    for config_name, cfg_data in ckpt["configs"].items():
        fname = f"walk_forward_results_{config_name.replace('+', '_')}.json"
        with open(RESULTS_DIR / fname, "w", encoding="utf-8") as f:
            json.dump(cfg_data["results"], f, ensure_ascii=False, indent=2)

    # 하위 호환성: walk_forward_results.json = AB+C 결과
    with open(RESULTS_DIR / "walk_forward_results.json", "w", encoding="utf-8") as f:
        json.dump(ckpt["configs"]["AB+C"]["results"], f, ensure_ascii=False, indent=2)

    _print_all_configs_summary(ckpt["configs"])
    return ckpt["configs"]


def _print_all_configs_summary(configs_results: dict) -> None:
    """4가지 구성 WFV 결과 요약 출력."""
    print("\n" + "=" * 60)
    print("4가지 구성 통합 백테스팅 결과")
    print("=" * 60)
    for config_name, cfg_data in configs_results.items():
        results = cfg_data["results"]
        if not results:
            continue
        pass_count = sum(1 for r in results if r.get("beats_benchmark", False))
        print(f"\n[{config_name}] — {pass_count}/{len(results)} fold KOSPI200 초과")
        print(f"  {'연도':<6} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'벤치초과':>6}")
        for r in results:
            if "error" in r:
                print(f"  {r.get('test_year','?')}: 오류 — {r['error']}")
                continue
            print(
                f"  {r.get('test_year','?'):<6} "
                f"{r.get('cagr',0)*100:>7.2f}% "
                f"{r.get('sharpe',0):>8.2f} "
                f"{r.get('mdd',0)*100:>7.2f}% "
                f"{'✓' if r.get('beats_benchmark') else '✗':>6}"
            )
    print(f"\n합격 기준: {WFV_CONFIG['pass_threshold']}/5 fold KOSPI200 초과")


if __name__ == "__main__":
    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    if mode in ("A", "B"):
        run_ab_wfv(mode)
    elif mode == "ALL":
        run_all_configs()
    else:
        result = run_full_wfv()
        print_wfv_summary()
