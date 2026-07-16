"""
Agent C — CNN-1D 단기 타점 모델.

입력: 원시 OHLCV (10영업일 윈도우 × 5채널)
출력: 3클래스 확률 [Buy=0, Hold=1, Sell=2]
라벨: ATR hit-target (N=10, K=1.0)

설계 원칙:
  - LSTM과 동일한 가공 지표 입력 금지 (원시 OHLCV만)
  - 윈도우 내 종가 기준 정규화 (스케일 통일)
  - CrossEntropyLoss (회귀 방식 금지)
  - bidirectional 없음
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config.settings import CNN_CONFIG, LABEL_CONFIG, MODELS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

SAVED_DIR = MODELS_DIR / "saved"

# 라벨 인덱스
BUY  = 0
HOLD = 1
SELL = 2

N_CLASSES  = 3
N_CHANNELS = len(CNN_CONFIG["channels"])  # 5 (OHLCV)
WINDOW     = CNN_CONFIG["window"]         # 10


class CNN1DDataset(Dataset):
    """
    CNN-1D 학습 데이터셋.

    X : (샘플 수, 채널 수=5, 윈도우) float32
    y : (샘플 수,) long (0=Buy, 1=Hold, 2=Sell)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, window: int = WINDOW):
        assert X.shape[1:] == (N_CHANNELS, window), (
            f"X shape 오류: 기대 (N, {N_CHANNELS}, {window}), 실제 {X.shape}"
        )
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1DModel(nn.Module):
    """
    CNN-1D 아키텍처.
    WFV 전 소규모 탐색으로 최적 구조 결정 후 이 클래스를 수정.
    기본값: 2 Conv 레이어, 64 필터, 커널 3
    """

    def __init__(
        self,
        n_channels:      int = N_CHANNELS,
        window:          int = WINDOW,
        num_conv_layers: int = 2,
        num_filters:     int = 64,
        kernel_size:     int = 3,
        dropout:         float = 0.2,
        n_classes:       int = N_CLASSES,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_ch = n_channels
        for _ in range(num_conv_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, num_filters, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            in_ch = num_filters

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(num_filters, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.conv_layers:
            x = layer(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        x = x.to(device)
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def signal_score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CNN_score = (P(Buy) - P(Sell)) × (1 - H(softmax))

        Returns
        -------
        (signal, confidence)  — 각 배치 행 기준
        """
        proba = self.predict_proba(x)
        signal = proba[:, BUY] - proba[:, SELL]

        # 엔트로피 기반 신뢰도
        eps  = 1e-9
        h    = -(proba * (proba + eps).log()).sum(dim=-1) / np.log(N_CLASSES)
        conf = 1.0 - h

        return signal * conf, conf

    def signal_score_bin(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        이진 모드 신호: P(Buy=class1) × (1 - H)
        n_classes=2 모델 전용 (BUY=1, NOBUY=0)
        """
        proba = self.predict_proba(x)          # (B, 2)
        signal = proba[:, 1]                   # P(Buy), range [0, 1]
        eps  = 1e-9
        h    = -(proba * (proba + eps).log()).sum(dim=-1) / np.log(2)
        conf = 1.0 - h
        return signal * conf, conf


# ── 라벨 생성 ─────────────────────────────────────────────────────────────────

def generate_labels_bin(
    price_df: pd.DataFrame,
    K: float = 2.0,
    N: int = 10,
) -> pd.DataFrame:
    """
    이진 라벨 생성: N일 내 +K×ATR 도달 여부 (1=도달, 0=미도달).
    경로 중 하락 여부는 무시 — "상방 힘이 있는가"만 판정.
    """
    df = price_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    tr    = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    )
    tr  = np.concatenate([[tr[0]], tr])
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    labels = np.zeros(len(df), dtype=np.int64)  # 기본 0 (NoBuy)

    for i in range(len(df) - N):
        target_up = close[i] + K * atr[i]
        for j in range(1, N + 1):
            if i + j >= len(df):
                break
            if high[i + j] >= target_up:
                labels[i] = 1  # Buy
                break

    df["label"] = labels
    return df[["Date", "label"]]


def generate_labels(
    price_df: pd.DataFrame,
    ref_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    ATR hit-target 라벨 생성 (N=10, K=1.0).

    Returns
    -------
    DataFrame with columns: Date, label (0=Buy,1=Hold,2=Sell)
    """
    df = price_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    N = LABEL_CONFIG["N"]
    K = LABEL_CONFIG["K"]

    # ATR (14일 이동 평균)
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    tr    = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    )
    tr = np.concatenate([[tr[0]], tr])
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    labels = np.full(len(df), HOLD, dtype=np.int64)

    for i in range(len(df) - N):
        c = close[i]
        a = atr[i]
        target_up   = c + K * a
        target_down = c - K * a

        hit_up   = False
        hit_down = False
        for j in range(1, N + 1):
            if i + j >= len(df):
                break
            if high[i + j] >= target_up:
                hit_up = True
            if low[i + j] <= target_down:
                hit_down = True

        if hit_up and not hit_down:
            labels[i] = BUY
        elif hit_down and not hit_up:
            labels[i] = SELL
        else:
            labels[i] = HOLD

    df["label"] = labels
    return df[["Date", "label"]]


# ── 입력 시퀀스 생성 ──────────────────────────────────────────────────────────

def make_sequences(
    price_df: pd.DataFrame,
    label_df: pd.DataFrame,
    window: int = WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """
    (X, y) 시퀀스 생성.

    X shape: (N, 5, window)  — CNN 입력 (채널 우선)
    y shape: (N,)             — 라벨

    정규화: 윈도우 내 종가 기준 (마지막 종가 = 1.0)
    window 기본값 = WINDOW(10), 100일 등 다른 값도 허용.
    """
    df = price_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    label_map = label_df.set_index("Date")["label"].to_dict()

    xs, ys = [], []

    for i in range(window, len(df)):
        date = df.iloc[i]["Date"]
        if date not in label_map:
            continue
        label = label_map[date]

        window_df = df.iloc[i - window: i]
        ref_close = window_df.iloc[-1]["Close"]
        if ref_close == 0:
            continue

        # 정규화 (종가 기준)
        o = window_df["Open"].values  / ref_close
        h = window_df["High"].values  / ref_close
        l = window_df["Low"].values   / ref_close
        c = window_df["Close"].values / ref_close
        # 거래량: 윈도우 내 최대 거래량 기준
        v_max = window_df["Volume"].max()
        v = window_df["Volume"].values / (v_max + 1e-9)

        x = np.stack([o, h, l, c, v], axis=0).astype(np.float32)  # (5, window)
        xs.append(x)
        ys.append(int(label))

    return np.array(xs), np.array(ys, dtype=np.int64)


# ── 학습 ─────────────────────────────────────────────────────────────────────

def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    arch_params:     dict | None = None,
    epochs:          int  = 200,
    batch_size:      int  = 256,
    lr:              float = 1e-3,
    device_str:      str  = "cpu",
    checkpoint_path: str | None = None,
    checkpoint_every: int = 25,
) -> CNN1DModel:
    """
    CNN-1D 학습.

    Parameters
    ----------
    arch_params      : CNN_ARCH_SEARCH 범위 내 파라미터 (None이면 기본값)
    checkpoint_path  : 체크포인트 저장 경로 (.ckpt). None이면 저장 안 함.
    checkpoint_every : N 에폭마다 체크포인트 저장 (기본 25)
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    if arch_params is None:
        arch_params = {}

    model = CNN1DModel(
        num_conv_layers=arch_params.get("num_conv_layers", 2),
        num_filters=arch_params.get("num_filters", 64),
        kernel_size=arch_params.get("kernel_size", 3),
        dropout=arch_params.get("dropout", 0.2),
    ).to(device)

    dataset    = CNN1DDataset(X_train, y_train, window=X_train.shape[2])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.CrossEntropyLoss()

    start_epoch = 1
    ckpt_path   = Path(checkpoint_path) if checkpoint_path else None

    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info("CNN 체크포인트 로드 — Epoch %d부터 재개 (%s)", start_epoch, ckpt_path.name)

    model.train()
    for epoch in range(start_epoch, epochs + 1):
        total_loss = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        avg_loss = total_loss / len(dataset)

        if epoch % 10 == 0 or epoch == epochs:
            logger.info("CNN Epoch %d/%d | loss=%.4f", epoch, epochs, avg_loss)
            if avg_loss < 0.5:
                logger.info("Loss 충분히 수렴 — 조기 종료 가능")

        if ckpt_path and epoch % checkpoint_every == 0:
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)

    if ckpt_path and ckpt_path.exists():
        ckpt_path.unlink()

    return model


def save_cnn(model: CNN1DModel, path: str | None = None) -> None:
    p = SAVED_DIR / "cnn_signal_c.pt" if path is None else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    logger.info("CNN 모델 저장: %s", p)


def load_cnn(arch_params: dict | None = None, path: str | None = None) -> CNN1DModel | None:
    p = SAVED_DIR / "cnn_signal_c.pt" if path is None else Path(path)
    if not p.exists():
        return None
    if arch_params is None:
        arch_params = {}
    model = CNN1DModel(
        num_conv_layers=arch_params.get("num_conv_layers", 2),
        num_filters=arch_params.get("num_filters", 64),
        kernel_size=arch_params.get("kernel_size", 3),
        dropout=arch_params.get("dropout", 0.2),
    )
    model.load_state_dict(torch.load(p, map_location="cpu"))
    model.eval()
    return model


