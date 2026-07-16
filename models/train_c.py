"""
Agent C CNN-1D + LSTM 학습 파이프라인.

1. 전 종목 OHLCV에서 ATR hit-target 라벨 생성 (N=10, K=1.0)
2. WFV 확장 윈도우 구조로 CNN/LSTM 학습
3. WFV 전 소규모 아키텍처 탐색 (fold 1 기준)
4. 모델 저장 및 신호 DataFrame 생성

실행: python -m models.train_c [arch_search|wfv|full]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config.settings import (
    ALPHA_GRID,
    BACKTEST_DIR,
    CNN_ARCH_SEARCH,
    LSTM_ARCH_SEARCH,
    MODELS_DIR,
    PRICE_START_DATE,
    WFV_CONFIG,
)
from data.build_price_cache import load_price_cache
from data.build_universe import get_universe_by_date
from models.cnn_model import (
    CNN1DModel,
    CNN1DDataset,
    generate_labels,
    generate_labels_bin,
    load_cnn,
    make_sequences,
    save_cnn,
    train_cnn,
    WINDOW as CNN_WINDOW,
)
from models.lstm_model import (
    LSTMModel,
    LSTMDataset,
    compute_technical_features,
    load_lstm,
    make_lstm_sequences,
    save_lstm,
    train_lstm,
    WINDOW as LSTM_WINDOW,
)
from utils.calendar_utils import get_trading_days
from utils.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = BACKTEST_DIR / "results"
SAVED_DIR   = MODELS_DIR / "saved"
ARCHIVE_DIR = MODELS_DIR / "archive"


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── 전 종목 데이터 수집 ────────────────────────────────────────────────────────

def collect_all_ohlcv(
    tickers: list[str],
    start_year: int,
    end_year: int,
    label_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    전 종목 합산 CNN/LSTM 학습 데이터 수집.

    Returns
    -------
    (X_cnn, y_cnn, X_lstm, y_lstm)
    """
    if label_fn is None:
        label_fn = generate_labels

    start_str = f"{start_year}-01-01"
    end_str   = f"{end_year}-12-31"

    all_X_cnn:  list[np.ndarray] = []
    all_y_cnn:  list[np.ndarray] = []
    all_X_lstm: list[np.ndarray] = []
    all_y_lstm: list[np.ndarray] = []

    for ticker in tqdm(tickers, desc="OHLCV 수집"):
        df = load_price_cache(ticker)
        if df is None or df.empty:
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[(df["Date"] >= start_str) & (df["Date"] <= end_str)].copy()
        if len(df) < LSTM_WINDOW + 20:
            continue

        # 라벨 생성 (CNN·LSTM 공통)
        label_df = label_fn(df)

        # CNN 시퀀스
        X_cnn, y_cnn = make_sequences(df, label_df)
        if len(X_cnn) > 0:
            all_X_cnn.append(X_cnn)
            all_y_cnn.append(y_cnn)

        # LSTM 기술적 지표 계산 후 시퀀스
        try:
            tech_df = compute_technical_features(df)
            X_lstm, y_lstm = make_lstm_sequences(tech_df, label_df)
            if len(X_lstm) > 0:
                all_X_lstm.append(X_lstm)
                all_y_lstm.append(y_lstm)
        except AssertionError as e:
            logger.warning("%s 기술 지표 계산 실패: %s", ticker, e)

    X_cnn  = np.concatenate(all_X_cnn,  axis=0) if all_X_cnn  else np.empty((0, 5, CNN_WINDOW))
    y_cnn  = np.concatenate(all_y_cnn,  axis=0) if all_y_cnn  else np.empty(0, dtype=np.int64)
    X_lstm = np.concatenate(all_X_lstm, axis=0) if all_X_lstm else np.empty((0, LSTM_WINDOW, 17))
    y_lstm = np.concatenate(all_y_lstm, axis=0) if all_y_lstm else np.empty(0, dtype=np.int64)

    logger.info(
        "데이터 수집 완료 — CNN: %d 샘플, LSTM: %d 샘플",
        len(X_cnn), len(X_lstm),
    )
    # 클래스 분포 출력 (K=1.0 재실험 요구사항)
    for name, y in [("CNN", y_cnn), ("LSTM", y_lstm)]:
        if len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            dist = dict(zip(["Buy", "Hold", "Sell"], counts / counts.sum()))
            logger.info("%s 라벨 분포: %s", name, {k: f"{v:.2%}" for k, v in dist.items()})
            if dist.get("Buy", 0) > 0.50:
                logger.warning("Buy 비율 50%% 초과 — K(현재=1.0)를 1.2~1.5로 소폭 조정 검토")

    return X_cnn, y_cnn, X_lstm, y_lstm


# ── 아키텍처 탐색 ──────────────────────────────────────────────────────────────

def architecture_search(
    X_cnn_train: np.ndarray, y_cnn_train: np.ndarray,
    X_cnn_val:   np.ndarray, y_cnn_val:   np.ndarray,
    X_lstm_train: np.ndarray, y_lstm_train: np.ndarray,
    X_lstm_val:   np.ndarray, y_lstm_val:   np.ndarray,
    quick: bool = True,
) -> tuple[dict, dict]:
    """
    CNN·LSTM 소규모 아키텍처 탐색.
    quick=True이면 1 에폭만 훈련 후 검증셋 정확도로 순위 매김 (빠른 필터링).
    full run은 fold 1 결과 기준.

    Returns
    -------
    (best_cnn_params, best_lstm_params)
    """
    from sklearn.model_selection import ParameterGrid

    device = _get_device()
    epochs = 5 if quick else 50

    # CNN 탐색
    best_cnn_acc = -1.0
    best_cnn_params: dict = {}

    cnn_grid = {k: v for k, v in CNN_ARCH_SEARCH.items() if isinstance(v, list)}
    for params in ParameterGrid(cnn_grid):
        try:
            model = train_cnn(X_cnn_train, y_cnn_train, arch_params=params, epochs=epochs, device_str=device)
            model.eval()
            model_device = next(model.parameters()).device
            with torch.no_grad():
                X_t = torch.tensor(X_cnn_val, dtype=torch.float32).to(model_device)
                preds = model(X_t).argmax(dim=1).cpu().numpy()
            acc = (preds == y_cnn_val).mean()
            logger.debug("CNN %s → acc=%.4f", params, acc)
            if acc > best_cnn_acc:
                best_cnn_acc    = acc
                best_cnn_params = params.copy()
        except Exception as e:
            logger.debug("CNN 파라미터 실패 %s: %s", params, e)

    # LSTM 탐색
    best_lstm_acc = -1.0
    best_lstm_params: dict = {}

    lstm_grid = {k: v for k, v in LSTM_ARCH_SEARCH.items() if isinstance(v, list)}
    for params in ParameterGrid(lstm_grid):
        try:
            model = train_lstm(X_lstm_train, y_lstm_train, arch_params=params, epochs=epochs, device_str=device)
            model.eval()
            model_device = next(model.parameters()).device
            with torch.no_grad():
                X_t = torch.tensor(X_lstm_val, dtype=torch.float32).to(model_device)
                preds = model(X_t).argmax(dim=1).cpu().numpy()
            acc = (preds == y_lstm_val).mean()
            logger.debug("LSTM %s → acc=%.4f", params, acc)
            if acc > best_lstm_acc:
                best_lstm_acc    = acc
                best_lstm_params = params.copy()
        except Exception as e:
            logger.debug("LSTM 파라미터 실패 %s: %s", params, e)

    logger.info("최적 CNN 파라미터 (acc=%.4f): %s", best_cnn_acc, best_cnn_params)
    logger.info("최적 LSTM 파라미터 (acc=%.4f): %s", best_lstm_acc, best_lstm_params)

    # 저장
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    arch_result = {
        "cnn_best_params":  best_cnn_params,
        "lstm_best_params": best_lstm_params,
    }
    with open(SAVED_DIR / "arch_search_c.json", "w", encoding="utf-8") as f:
        json.dump(arch_result, f, ensure_ascii=False, indent=2)

    return best_cnn_params, best_lstm_params


def load_arch_params() -> tuple[dict, dict]:
    """저장된 아키텍처 탐색 결과 로드. 없으면 기본값 반환."""
    path = SAVED_DIR / "arch_search_c.json"
    if not path.exists():
        return {}, {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("cnn_best_params", {}), d.get("lstm_best_params", {})


# ── WFV 학습 ─────────────────────────────────────────────────────────────────

def run_c_wfv(cnn_params: dict | None = None, lstm_params: dict | None = None) -> list[dict]:
    """
    Agent C WFV 5-fold 학습.
    각 fold: 2014~train_end_year 학습 → test_year 신호 생성.

    체크포인트 (중단 후 재시작 지원):
    ① signals_c_fold{N}.parquet 존재 → fold 완전 완료, 스킵
    ② cnn/lstm_signal_c_fold{N}.pt 존재 → 학습 완료, 신호 생성만 재개
    """
    c_start = WFV_CONFIG["c_train_start"]  # 2014
    device  = _get_device()

    if cnn_params is None or lstm_params is None:
        cnn_params, lstm_params = load_arch_params()

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []

    for i, test_year in enumerate(WFV_CONFIG["test_years"]):
        sig_path       = RESULTS_DIR / f"signals_c_fold{i+1}.parquet"
        cnn_fold_path  = SAVED_DIR   / f"cnn_signal_c_fold{i+1}.pt"
        lstm_fold_path = SAVED_DIR   / f"lstm_signal_c_fold{i+1}.pt"

        # 체크포인트 ①: 신호 parquet 존재 → fold 완전 완료, 스킵
        if sig_path.exists():
            logger.info("Fold %d (%d) — 체크포인트 스킵 (%s 존재)",
                        i + 1, test_year, sig_path.name)
            all_metrics.append({
                "fold": i + 1, "test_year": test_year,
                "n_train_cnn": 0, "n_train_lstm": 0, "status": "checkpoint",
            })
            continue

        train_end = test_year - 1
        logger.info("=== Agent C Fold %d: %d~%d 학습 → %d 테스트 ===",
                    i + 1, c_start, train_end, test_year)

        # 학습 기간(c_start ~ train_end) 내 유니버스 합집합 (생존편향 방지)
        from data.build_universe import get_all_tickers_until
        tickers = get_all_tickers_until(train_end)
        if not tickers:
            logger.warning("유니버스 없음 — 스킵")
            continue

        # 체크포인트 ②: 모델 파일 존재 → 학습 스킵, 신호 생성만 재개
        cnn_model  = None
        lstm_model = None
        n_cnn = n_lstm = 0

        if cnn_fold_path.exists() and lstm_fold_path.exists():
            logger.info("Fold %d — 저장된 fold 모델 로드 (학습 스킵)", i + 1)
            cnn_model  = load_cnn(arch_params=cnn_params,  path=str(cnn_fold_path))
            lstm_model = load_lstm(arch_params=lstm_params, path=str(lstm_fold_path))
            if cnn_model is None or lstm_model is None:
                logger.warning("Fold %d 모델 로드 실패 → 재학습", i + 1)
                cnn_model  = None
                lstm_model = None

        if cnn_model is None:
            X_cnn, y_cnn, X_lstm, y_lstm = collect_all_ohlcv(tickers, c_start, train_end)
            if len(X_cnn) == 0 or len(X_lstm) == 0:
                logger.warning("Fold %d: 데이터 없음 — 스킵", i + 1)
                continue
            n_cnn  = len(X_cnn)
            n_lstm = len(X_lstm)
            cnn_model  = train_cnn(X_cnn,  y_cnn,  arch_params=cnn_params,  epochs=200, device_str=device)
            lstm_model = train_lstm(X_lstm, y_lstm, arch_params=lstm_params, epochs=200, device_str=device)
            # 학습 완료 즉시 저장 → 재시작 시 체크포인트 ②로 복원
            save_cnn(cnn_model,   str(cnn_fold_path))
            save_lstm(lstm_model, str(lstm_fold_path))
            logger.info("Fold %d 모델 저장 완료", i + 1)

        # 테스트 기간 신호 생성
        signals_df = generate_signals_for_period(
            cnn_model, lstm_model,
            tickers=tickers,
            start=f"{test_year}-01-01",
            end=f"{test_year}-12-31",
            alpha=0.5,  # fold 1 그리드서치로 나중에 결정
        )
        signals_df.to_parquet(sig_path, index=False)

        metrics = {
            "fold":         i + 1,
            "test_year":    test_year,
            "n_train_cnn":  n_cnn,
            "n_train_lstm": n_lstm,
        }
        all_metrics.append(metrics)
        logger.info("Fold %d 완료 — 신호 저장: %s", i + 1, sig_path)

    return all_metrics


def generate_signals_for_period(
    cnn_model:  CNN1DModel,
    lstm_model: LSTMModel,
    tickers:    list[str],
    start:      str,
    end:        str,
    alpha:      float = 0.5,
    bin_mode:   bool  = False,
) -> pd.DataFrame:
    """
    지정 기간 모든 종목의 daily CNN/LSTM 신호 생성.

    Returns
    -------
    DataFrame: date, ticker, cnn_score, lstm_score, final_score
    """
    beta    = 1.0 - alpha
    device  = _get_device()
    rows: list[dict] = []

    trading_days = get_trading_days(start, end)

    for ticker in tqdm(tickers, desc=f"신호 생성 {start[:4]}"):
        df = load_price_cache(ticker)
        if df is None or df.empty:
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Date"] <= end].copy()
        if len(df) < max(CNN_WINDOW, LSTM_WINDOW) + 5:
            continue

        try:
            tech_df = compute_technical_features(df)
        except AssertionError:
            continue

        cnn_model.eval()
        lstm_model.eval()

        for date in trading_days:
            # CNN 입력
            past_df = df[df["Date"] < date].tail(CNN_WINDOW)
            if len(past_df) < CNN_WINDOW:
                continue
            ref_close = float(past_df.iloc[-1]["Close"])
            if ref_close == 0:
                continue
            o = past_df["Open"].values   / ref_close
            h = past_df["High"].values   / ref_close
            l = past_df["Low"].values    / ref_close
            c = past_df["Close"].values  / ref_close
            v_max = past_df["Volume"].max()
            v = past_df["Volume"].values / (v_max + 1e-9)
            x_cnn = torch.tensor(np.stack([o, h, l, c, v])[np.newaxis], dtype=torch.float32)

            # LSTM 입력
            past_tech = tech_df[tech_df.index < date].tail(LSTM_WINDOW)
            if len(past_tech) < LSTM_WINDOW:
                continue
            x_lstm = torch.tensor(
                past_tech.values[np.newaxis].astype(np.float32), dtype=torch.float32
            )

            with torch.no_grad():
                if bin_mode:
                    cnn_sig,  _ = cnn_model.signal_score_bin(x_cnn)
                    lstm_sig, _ = lstm_model.signal_score_bin(x_lstm)
                else:
                    cnn_sig,  _ = cnn_model.signal_score(x_cnn)
                    lstm_sig, _ = lstm_model.signal_score(x_lstm)

            final = alpha * float(cnn_sig[0]) + beta * float(lstm_sig[0])
            rows.append({
                "date":        date.strftime("%Y-%m-%d"),
                "ticker":      ticker,
                "cnn_score":   round(float(cnn_sig[0]),  4),
                "lstm_score":  round(float(lstm_sig[0]), 4),
                "final_score": round(final, 4),
            })

    return pd.DataFrame(rows)


def run_c_wfv_bin(cnn_params: dict | None = None, lstm_params: dict | None = None) -> list[dict]:
    """
    Agent C 이진분류 WFV (K=2.0, n_classes=2).
    기존 ATR 3클래스 모델과 완전히 분리된 별도 파일 저장.

    체크포인트:
    ① signals_c_bin_fold{N}.parquet 존재 → 완전 완료, 스킵
    ② cnn/lstm_signal_c_bin_fold{N}.pt 존재 → 학습 스킵, 신호 생성만 재개
    """
    c_start = WFV_CONFIG["c_train_start"]
    device  = _get_device()

    if cnn_params is None or lstm_params is None:
        cnn_params, lstm_params = load_arch_params()

    # 이진 모드: n_classes=2
    cnn_params_bin  = {**cnn_params,  "n_classes": 2}
    lstm_params_bin = {**lstm_params, "n_classes": 2}

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []

    for i, test_year in enumerate(WFV_CONFIG["test_years"]):
        sig_path       = RESULTS_DIR / f"signals_c_bin_fold{i+1}.parquet"
        cnn_fold_path  = SAVED_DIR   / f"cnn_signal_c_bin_fold{i+1}.pt"
        lstm_fold_path = SAVED_DIR   / f"lstm_signal_c_bin_fold{i+1}.pt"

        if sig_path.exists():
            logger.info("Fold %d (%d) [bin] — 체크포인트 스킵 (%s 존재)",
                        i + 1, test_year, sig_path.name)
            all_metrics.append({"fold": i+1, "test_year": test_year, "status": "checkpoint"})
            continue

        train_end = test_year - 1
        logger.info("=== [BIN] Agent C Fold %d: %d~%d 학습 → %d 테스트 ===",
                    i + 1, c_start, train_end, test_year)

        from data.build_universe import get_all_tickers_until
        tickers = get_all_tickers_until(train_end)
        if not tickers:
            logger.warning("유니버스 없음 — 스킵")
            continue

        cnn_model  = None
        lstm_model = None
        n_cnn = n_lstm = 0

        if cnn_fold_path.exists() and lstm_fold_path.exists():
            logger.info("Fold %d [bin] — 저장된 fold 모델 로드 (학습 스킵)", i + 1)
            cnn_model  = load_cnn(arch_params=cnn_params_bin,  path=str(cnn_fold_path))
            lstm_model = load_lstm(arch_params=lstm_params_bin, path=str(lstm_fold_path))
            if cnn_model is None or lstm_model is None:
                cnn_model = lstm_model = None

        if cnn_model is None:
            X_cnn, y_cnn, X_lstm, y_lstm = collect_all_ohlcv(
                tickers, c_start, train_end, label_fn=generate_labels_bin
            )
            if len(X_cnn) == 0 or len(X_lstm) == 0:
                logger.warning("Fold %d [bin]: 데이터 없음 — 스킵", i + 1)
                continue

            # 라벨 분포 출력
            for name, y in [("CNN", y_cnn), ("LSTM", y_lstm)]:
                vals, counts = np.unique(y, return_counts=True)
                dist = {("Buy" if v == 1 else "NoBuy"): f"{c/len(y):.1%}" for v, c in zip(vals, counts)}
                logger.info("%s 라벨 분포: %s", name, dist)

            n_cnn, n_lstm = len(X_cnn), len(X_lstm)
            cnn_model  = train_cnn(X_cnn,  y_cnn,  arch_params=cnn_params_bin,  epochs=200, device_str=device)
            lstm_model = train_lstm(X_lstm, y_lstm, arch_params=lstm_params_bin, epochs=200, device_str=device)
            save_cnn(cnn_model,   str(cnn_fold_path))
            save_lstm(lstm_model, str(lstm_fold_path))
            logger.info("Fold %d [bin] 모델 저장 완료", i + 1)

        signals_df = generate_signals_for_period(
            cnn_model, lstm_model,
            tickers=tickers,
            start=f"{test_year}-01-01",
            end=f"{test_year}-12-31",
            alpha=0.5,
            bin_mode=True,
        )
        signals_df.to_parquet(sig_path, index=False)

        all_metrics.append({"fold": i+1, "test_year": test_year,
                             "n_train_cnn": n_cnn, "n_train_lstm": n_lstm})
        logger.info("Fold %d [bin] 완료 — 신호 저장: %s", i + 1, sig_path)

    return all_metrics


def train_full_c(cnn_params: dict | None = None, lstm_params: dict | None = None) -> None:
    """
    전기간 재학습 (실운용 모델).
    WFV pass_threshold 통과 확인 후에만 호출.
    """
    if cnn_params is None or lstm_params is None:
        cnn_params, lstm_params = load_arch_params()

    c_start = WFV_CONFIG["c_train_start"]
    end_year = pd.Timestamp.today().year

    # 전체 역사적 유니버스 합집합 (생존편향 방지)
    from data.build_universe import get_all_tickers_until
    tickers = get_all_tickers_until(end_year)
    if not tickers:
        logger.error("전기간 재학습 실패 (유니버스 없음)")
        return

    X_cnn, y_cnn, X_lstm, y_lstm = collect_all_ohlcv(tickers, c_start, end_year)

    if len(X_cnn) == 0:
        logger.error("CNN 학습 데이터 없음")
        return

    device = _get_device()
    cnn_model  = train_cnn(X_cnn, y_cnn, arch_params=cnn_params, epochs=200, device_str=device)
    lstm_model = train_lstm(X_lstm, y_lstm, arch_params=lstm_params, epochs=200, device_str=device)

    # 기존 모델 백업
    from datetime import datetime
    stamp = datetime.today().strftime("%Y%m%d")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ["cnn_signal_c.pt", "lstm_signal_c.pt"]:
        src = SAVED_DIR / fname
        if src.exists():
            import shutil
            shutil.copy(src, ARCHIVE_DIR / f"{fname[:-3]}_{stamp}.pt")

    save_cnn(cnn_model)
    save_lstm(lstm_model)
    logger.info("Agent C 전기간 재학습 완료")


def check_model_status() -> bool:
    """
    모델 로드 상태 체크 및 로그 출력.
    미탑재 시 0.0 폴백 모드 명시.
    """
    cnn_loaded  = (SAVED_DIR / "cnn_signal_c.pt").exists()
    lstm_loaded = (SAVED_DIR / "lstm_signal_c.pt").exists()

    if not cnn_loaded or not lstm_loaded:
        logger.warning(
            "CNN %s | LSTM %s — 0.0 폴백 모드로 실행 중 (거래 비활성)",
            "✓" if cnn_loaded else "✗",
            "✓" if lstm_loaded else "✗",
        )
    else:
        logger.info("CNN ✓ | LSTM ✓ — 모델 정상 로드")
    return cnn_loaded and lstm_loaded


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "wfv"

    if mode == "arch_search":
        logger.info("아키텍처 탐색 모드")
        tickers = get_universe_by_date(pd.Timestamp.today().strftime("%Y-%m-%d"))
        X_cnn, y_cnn, X_lstm, y_lstm = collect_all_ohlcv(
            tickers, WFV_CONFIG["c_train_start"], WFV_CONFIG["test_years"][0] - 1
        )
        # fold 1 학습/검증 분할 (80:20)
        n = len(X_cnn)
        split = int(n * 0.8)
        architecture_search(
            X_cnn[:split], y_cnn[:split], X_cnn[split:], y_cnn[split:],
            X_lstm[:split], y_lstm[:split], X_lstm[split:], y_lstm[split:],
        )
    elif mode == "full":
        train_full_c()
    elif mode == "wfv_bin":
        run_c_wfv_bin()
    else:
        run_c_wfv()
