"""
Agent C — LSTM 중기 맥락 모델.

입력: 17개 기술적 지표 (20영업일 윈도우)
출력: 3클래스 확률 [Buy=0, Hold=1, Sell=2]
라벨: ATR hit-target (N=10, K=1.0) — CNN과 동일

설계 원칙:
  - CrossEntropyLoss (회귀 MSELoss 금지)
  - bidirectional=False (미래 데이터 누수 차단)
  - 최소 200 에폭 (loss > ln(3)=1.099 이면 학습 실패)
  - atr_norm 피처 제외 (라벨 계산에 ATR 사용 → 누수 위험)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config.settings import LABEL_CONFIG, LSTM_CONFIG, LSTM_FEATURES, MODELS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

SAVED_DIR  = MODELS_DIR / "saved"
N_CLASSES  = LSTM_CONFIG["n_classes"]   # 3
WINDOW     = LSTM_CONFIG["window"]       # 20
N_FEATURES = LSTM_CONFIG["n_features"]  # 17

BUY  = 0
HOLD = 1
SELL = 2


class LSTMDataset(Dataset):
    """
    X : (N, WINDOW, N_FEATURES) float32
    y : (N,) long
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, window: int = WINDOW):
        assert X.shape[1:] == (window, N_FEATURES), (
            f"X shape 오류: 기대 (N, {window}, {N_FEATURES}), 실제 {X.shape}"
        )
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """
    LSTM 아키텍처.
    기본값: 1 레이어, hidden_size=64, dropout=0.2
    WFV 전 소규모 탐색으로 최적 구조 결정 후 파라미터 교체.
    """

    def __init__(
        self,
        n_features:   int   = N_FEATURES,
        hidden_size:  int   = 64,
        num_layers:   int   = 1,
        dropout:      float = 0.2,
        n_classes:    int   = N_CLASSES,
        bidirectional: bool = False,  # 미래 누수 차단 — False 고정
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # 설정값 무시하고 항상 단방향
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])  # 마지막 타임스텝
        return self.fc(out)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        x = x.to(device)
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=-1)

    def signal_score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        LSTM_score = (P(Buy) - P(Sell)) × (1 - H(softmax))
        """
        proba  = self.predict_proba(x)
        signal = proba[:, BUY] - proba[:, SELL]
        eps    = 1e-9
        h      = -(proba * (proba + eps).log()).sum(dim=-1) / np.log(N_CLASSES)
        conf   = 1.0 - h
        return signal * conf, conf

    def signal_score_bin(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        이진 모드 신호: P(Buy=class1) × (1 - H)
        n_classes=2 모델 전용 (BUY=1, NOBUY=0)
        """
        proba  = self.predict_proba(x)          # (B, 2)
        signal = proba[:, 1]                    # P(Buy), range [0, 1]
        eps    = 1e-9
        h      = -(proba * (proba + eps).log()).sum(dim=-1) / np.log(2)
        conf   = 1.0 - h
        return signal * conf, conf


# ── 기술적 지표 계산 ──────────────────────────────────────────────────────────

def compute_technical_features(
    price_df: pd.DataFrame,
    sector_prices_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    LSTM_FEATURES 17개 계산.
    price_df: OHLCV DataFrame (Date 인덱스 또는 컬럼)

    반환: Date 인덱스, 17개 피처 컬럼
    """
    df = price_df.copy()
    if "Date" in df.columns:
        df = df.set_index("Date")
    df = df.sort_index()

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    open_  = df["Open"]

    out = pd.DataFrame(index=df.index)

    # ── 추세 방향 ──────────────────────────────────────────────────────────────

    ma5  = close.rolling(5,  min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    ma60 = close.rolling(60, min_periods=1).mean()
    # ma_alignment: 5>20>60 이면 1.0, 혼합이면 중간값
    align = ((close > ma5).astype(float) + (ma5 > ma20).astype(float) +
             (ma20 > ma60).astype(float)) / 3.0
    out["ma_alignment"] = align

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    out["macd_score"] = (macd - signal_line).clip(-1, 1)  # 히스토그램 방향

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    out["close_vs_ema20"] = (close / (ema20 + 1e-9) - 1).clip(-0.3, 0.3)
    out["close_vs_ema60"] = (close / (ema60 + 1e-9) - 1).clip(-0.5, 0.5)

    # ── 과매수/과매도 ──────────────────────────────────────────────────────────

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    rsi   = 100 - 100 / (1 + rs)
    out["rsi_norm"] = rsi / 100.0

    # 스토캐스틱
    lo14 = low.rolling(14).min()
    hi14 = high.rolling(14).max()
    out["stoch_norm"] = ((close - lo14) / (hi14 - lo14 + 1e-9)).clip(0, 1)

    # 볼린저 밴드 위치
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std
    bb_pos = (close - bb_dn) / (bb_up - bb_dn + 1e-9)
    out["bb_pos"] = bb_pos.clip(0, 1)

    # ── 독립 신호 ──────────────────────────────────────────────────────────────

    # ADX
    tr     = pd.concat([high - low,
                        (high - close.shift()).abs(),
                        (low  - close.shift()).abs()], axis=1).max(axis=1)
    dm_pos = (high - high.shift()).clip(lower=0)
    dm_neg = (low.shift() - low).clip(lower=0)
    atr14  = tr.rolling(14).mean()
    di_pos = 100 * dm_pos.rolling(14).mean() / (atr14 + 1e-9)
    di_neg = 100 * dm_neg.rolling(14).mean() / (atr14 + 1e-9)
    dx     = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-9)
    adx    = dx.rolling(14).mean()
    out["adx_norm"] = (adx / 100.0).clip(0, 1)

    # 거래량 비율
    vol_ma20 = volume.rolling(20).mean()
    out["vol_ratio"] = (volume / (vol_ma20 + 1e-9)).clip(0, 5) / 5.0

    # OBV 정규화
    obv = (volume * np.sign(close.diff().fillna(0))).cumsum()
    obv_norm = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-9)
    out["obv_norm"] = obv_norm.clip(-3, 3) / 3.0

    # ── 장기 맥락 ──────────────────────────────────────────────────────────────

    # 200일선 대비 이격률
    ma200 = close.rolling(200, min_periods=50).mean()
    out["close_vs_ma200"] = (close / (ma200 + 1e-9) - 1).clip(-0.5, 0.5)

    # 52주 고저 위치
    hi52 = high.rolling(252, min_periods=50).max()
    lo52 = low.rolling(252, min_periods=50).min()
    out["high52w_pos"] = ((close - lo52) / (hi52 - lo52 + 1e-9)).clip(0, 1)

    # 주봉 방향 (5거래일 수익률)
    out["week_return"] = close.pct_change(5).clip(-0.2, 0.2)

    # ATR / 볼린저 밴드 폭 (변동성 수축·팽창)
    bb_width = (bb_up - bb_dn) / (bb_mid + 1e-9)
    out["price_vs_bb_width"] = (atr14 / (bb_width * close + 1e-9)).clip(0, 2) / 2.0

    # VPT (거래량 × 가격변화율 누적)
    vpt = (volume * close.pct_change().fillna(0)).cumsum()
    vpt_norm = (vpt - vpt.rolling(20).mean()) / (vpt.rolling(20).std() + 1e-9)
    out["volume_price_trend"] = vpt_norm.clip(-3, 3) / 3.0

    # ── 신규 추가 피처 ──────────────────────────────────────────────────────────

    # 갭 비율
    out["gap_ratio"] = ((open_ - close.shift()) / (close.shift() + 1e-9)).clip(-0.1, 0.1)

    # 섹터 대비 상대 강도 (섹터 가격 데이터 없으면 0)
    if sector_prices_df is not None and not sector_prices_df.empty:
        sec_close = sector_prices_df["Close"] if "Close" in sector_prices_df.columns else None
        if sec_close is not None:
            sec_ret   = sec_close.pct_change(5).reindex(df.index).fillna(0)
            stock_ret = close.pct_change(5)
            out["sector_relative_strength"] = (stock_ret - sec_ret).clip(-0.2, 0.2)
        else:
            out["sector_relative_strength"] = 0.0
    else:
        out["sector_relative_strength"] = 0.0

    # NaN 채우기 (앞부분 윈도우 부족)
    out = out.ffill().fillna(0.0)

    # 피처 순서 일치 확인
    assert list(out.columns) == LSTM_FEATURES, (
        f"피처 순서 불일치. 기대: {LSTM_FEATURES}, 실제: {list(out.columns)}"
    )
    return out


def make_lstm_sequences(
    tech_df: pd.DataFrame,
    label_df: pd.DataFrame,
    window: int = WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """
    (X, y) 시퀀스 생성.

    X shape: (N, window, N_FEATURES)
    y shape: (N,)
    window 기본값 = WINDOW(20), 100일 등 다른 값도 허용.
    """
    label_map = label_df.set_index("Date")["label"].to_dict()
    xs, ys = [], []

    idx = tech_df.index
    feat_vals = tech_df[LSTM_FEATURES].values

    for i in range(window, len(idx)):
        date = idx[i]
        if date not in label_map:
            continue
        seq = feat_vals[i - window: i].astype(np.float32)
        xs.append(seq)
        ys.append(int(label_map[date]))

    return np.array(xs), np.array(ys, dtype=np.int64)


# ── 학습 ─────────────────────────────────────────────────────────────────────

def train_lstm(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    arch_params:      dict | None = None,
    epochs:           int   = 200,
    batch_size:       int   = 256,
    lr:               float = 1e-3,
    device_str:       str   = "cpu",
    checkpoint_path:  str | None = None,
    checkpoint_every: int = 25,
) -> LSTMModel:
    """LSTM 학습. 최소 200 에폭, loss > ln(3) 이면 경고.

    Parameters
    ----------
    checkpoint_path  : 체크포인트 저장 경로 (.ckpt). None이면 저장 안 함.
    checkpoint_every : N 에폭마다 체크포인트 저장 (기본 25)
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    if arch_params is None:
        arch_params = {}

    model = LSTMModel(
        n_features=arch_params.get("n_features", N_FEATURES),
        hidden_size=arch_params.get("hidden_size", 64),
        num_layers=arch_params.get("num_layers", 1),
        dropout=arch_params.get("dropout", 0.2),
    ).to(device)

    dataset    = LSTMDataset(X_train, y_train, window=X_train.shape[1])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.CrossEntropyLoss()

    start_epoch = 1
    final_loss  = np.log(N_CLASSES)
    ckpt_path   = Path(checkpoint_path) if checkpoint_path else None

    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info("LSTM 체크포인트 로드 — Epoch %d부터 재개 (%s)", start_epoch, ckpt_path.name)

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

        final_loss = total_loss / len(dataset)
        if epoch % 10 == 0 or epoch == epochs:
            logger.info("LSTM Epoch %d/%d | loss=%.4f (ln(3)=1.099)", epoch, epochs, final_loss)

        if ckpt_path and epoch % checkpoint_every == 0:
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)

    if ckpt_path and ckpt_path.exists():
        ckpt_path.unlink()

    # 학습 실패 경고
    ln3 = np.log(N_CLASSES)
    if final_loss >= ln3 * 0.98:
        logger.warning(
            "LSTM 학습 실패 의심: 최종 loss=%.4f ≈ ln(3)=%.4f. 에폭 수 증가 권장.", final_loss, ln3
        )

    return model


def save_lstm(model: LSTMModel, path: str | None = None) -> None:
    p = SAVED_DIR / "lstm_signal_c.pt" if path is None else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    logger.info("LSTM 모델 저장: %s", p)


def load_lstm(arch_params: dict | None = None, path: str | None = None) -> LSTMModel | None:
    p = SAVED_DIR / "lstm_signal_c.pt" if path is None else Path(path)
    if not p.exists():
        return None
    if arch_params is None:
        arch_params = {}
    model = LSTMModel(
        hidden_size=arch_params.get("hidden_size", 64),
        num_layers=arch_params.get("num_layers", 1),
        dropout=arch_params.get("dropout", 0.2),
    )
    model.load_state_dict(torch.load(p, map_location="cpu"))
    model.eval()
    return model
