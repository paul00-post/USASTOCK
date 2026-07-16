"""
에이전트 C — CNN-1D + LSTM 병렬 신호 기반 매매 타점 결정 + 한국투자증권(KIS) API 주문.

[확정 실운용 설정]
  pool: XGBoost B 상위 30종목 (전체 KOSPI200 유니버스 동적 스코어링)
  진입 임계값: final_score >= 0.05 → 5% 포지션
               final_score >= 0.075 → 11% 포지션
  no_partial_sell: 현금 부족 시 매수 스킵 (강제 청산 없음)
  alpha: 0.5 (CNN:LSTM = 50:50)
  모델 윈도우: 100영업일 (cnn_signal_c.pt / lstm_signal_c.pt)

[장중 실행 구조 — KRX 본장(09:00~15:30)만 사용]
  08:00        run_daily_signal()          — 전일 종가 기준 매수 후보·비중 계산
  09:01        run_market_open_entry()     — 실시간 시가 확인 + 갭 필터 + 매수 /
                                              보유 종목 익절 지정가 매도 주문 재등록
  09:00~15:19  check_intraday_stop_loss()  — 1분마다 분봉 조회해 손절선 스침 감시 (15:20~15:30은 동시호가라 제외)
  15:10        run_end_of_day_settlement() — 타이머(10영업일)·신호청산(4일 연속) 판정 + 첫 매도 시도
  15:10~15:19  check_eod_exit_retry()      — 1분마다 위 매도의 미체결/부분체결 잔량 재시도

  매수는 현재가+N틱, 매도(손절·타이머)는 현재가-N틱 지정가로 즉시 체결 보장.
  익절은 진입 시 정해진 가격에 지정가 매도를 걸어두되, 장마감 시 미체결 주문이
  자동취소되는 증권사 관행 때문에 매일 아침(09:01) 다시 걸어야 한다.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config.settings import (
    AGENT_B_FEATURES,
    BACKTEST_DIR,
    BASE_DIR,
    INITIAL_CAPITAL,
    KIWOOM_CONFIG,
    MODELS_DIR,
    RISK_LIMITS,
    SIGNAL_THRESHOLDS,
    TRADE_CONFIG,
    TRANSACTION_COSTS,
    TRADING_HALT_ALERT_DAYS,
)
from data.build_price_cache import load_price_cache
from data.build_universe import get_universe_by_date
from data.sector_classifier import load_sector_map
from models.cnn_model import CNN1DModel, load_cnn
from models.lstm_model import LSTMModel, compute_technical_features, load_lstm
from utils.calendar_utils import add_trading_days, count_trading_days
from utils.kis_client import get_account_balance, get_current_price, get_minute_candles
from utils.tick_size import price_minus_ticks, price_plus_ticks, round_to_tick
from utils.logger import get_logger

logger = get_logger(__name__)

PORTFOLIO_PATH = BASE_DIR / "portfolio_state.json"
REPORTS_DIR    = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

_MAX_HOLD = TRADE_CONFIG["max_hold_days"]
_GAP_WIN  = TRADE_CONFIG["gap_atr_window"]
_GAP_MULT = TRADE_CONFIG["gap_filter_mult"]
_SL_WIN   = TRADE_CONFIG["swing_low_window"]
_SL_BUF   = TRADE_CONFIG["swing_low_buffer"]
_TP_RR    = TRADE_CONFIG["take_profit_rr"]
_BUY_TICKS  = TRADE_CONFIG["buy_tick_offset"]
_SELL_TICKS = TRADE_CONFIG["sell_tick_offset"]

_COMM = TRANSACTION_COSTS["commission"]
_TAX  = TRANSACTION_COSTS["tax"]
_SLIP = TRANSACTION_COSTS["slippage"]

# ── 확정 실운용 C-setting ───────────────────────────────────────────────────────
POOL_N            = 30      # Agent B 상위 종목 수
BUY_THRESHOLD     = 0.05    # 진입 하한 임계값
SUPER_BUY_THRESHOLD = 0.075 # 11% 포지션 임계값
BUY_POSITION_PCT  = 0.05    # 5% 포지션
SUPER_BUY_PCT     = 0.11    # 11% 포지션

# 모델 윈도우 (학습 시 사용한 창 크기)
WINDOW_100D = 100


# ── 포트폴리오 상태 관리 ──────────────────────────────────────────────────────

def _load_portfolio() -> dict:
    if PORTFOLIO_PATH.exists():
        state = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
        state.setdefault("pending_buys", {})
        return state
    return {"holdings": {}, "pending_buys": {}, "cash": INITIAL_CAPITAL, "last_updated": None}


def _save_portfolio(state: dict) -> None:
    state["last_updated"] = datetime.now().isoformat()
    PORTFOLIO_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Agent B Pool 선택 ──────────────────────────────────────────────────────────

def _get_top30_pool(ref_date: pd.Timestamp) -> list[str]:
    """
    XGBoost B로 전체 KOSPI200 유니버스 스코어링 → 상위 POOL_N 종목 반환.
    factor_dataset_B.parquet에서 ref_date 이전 가장 최근 스냅샷 사용.
    """
    ds_path = BACKTEST_DIR / "results" / "factor_dataset_B.parquet"
    model_path = MODELS_DIR / "saved" / "xgb_agent_B_live.pkl"

    if not ds_path.exists():
        logger.warning("factor_dataset_B.parquet 없음 — 빈 풀 반환")
        return []
    if not model_path.exists():
        logger.warning("xgb_agent_B_live.pkl 없음 — 빈 풀 반환")
        return []

    try:
        df = pd.read_parquet(ds_path)
        df["date"] = pd.to_datetime(df["date"])

        # ref_date 이전 데이터만 (룩어헤드 방지)
        df = df[df["date"] < ref_date]
        if df.empty:
            logger.warning("factor_dataset_B: ref_date 이전 데이터 없음")
            return []

        # 현재 코스피200 유니버스로 제한 — factor_dataset_B.parquet에는 과거에
        # 유니버스였다가 지금은 빠진 종목의 스냅샷도 그대로 남아있어서, 이 필터가
        # 없으면 이미 유니버스에서 제외된 종목을 오래된 데이터로 신규 매수할 수 있었다
        # (agent_b.py의 watchlist 계산과도 이 부분이 어긋나는 원인이었음).
        try:
            universe = set(get_universe_by_date(ref_date.strftime("%Y-%m-%d")))
            df = df[df["ticker"].isin(universe)]
        except RuntimeError as e:
            logger.warning("유니버스 로드 실패 — 필터 없이 진행: %s", e)

        if df.empty:
            logger.warning("factor_dataset_B: 유니버스 필터 후 데이터 없음")
            return []

        # 종목별 가장 최근 행
        latest = df.sort_values("date").groupby("ticker").tail(1).copy()

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        feats = [c for c in AGENT_B_FEATURES if c in latest.columns]
        # 학습(train_fold/train_full)이 raw NaN을 그대로 XGBoost에 넘겨 자체
        # missing-value 분기로 처리하게 하므로, 실운용 예측도 동일하게 raw NaN을
        # 넘겨야 한다. fillna(0)은 "결측"을 "진짜 0"으로 오인시켜 학습 때와 다른
        # 분기로 흘러가 실제 선정 종목이 달라지는 문제가 있었다 (2026-07-09 발견).
        X = latest[feats].values.astype("float32")
        latest = latest.copy()
        latest["xgb_score"] = model.predict(X)

        top_n = latest.nlargest(POOL_N, "xgb_score")["ticker"].tolist()
        logger.info("Agent B 풀: %d/%d종목 → 상위 %d 선택", len(latest), len(df["ticker"].unique()), len(top_n))
        return top_n

    except Exception as e:
        logger.error("Agent B 풀 선택 오류: %s", e)
        return []


# ── ATR 계산 (전일까지 로컬 캐시 기준 — 룩어헤드 없음) ────────────────────────

def _compute_atr(ticker: str, ref_date: pd.Timestamp, window: int = 10) -> float | None:
    df = load_price_cache(ticker)
    if df is None:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    past = df[df["Date"] < ref_date].tail(window + 1)
    if len(past) < 2:
        return None
    h = past["High"].values
    l = past["Low"].values
    c = past["Close"].values
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(tr.mean()) if len(tr) > 0 else None


def _swing_low(ticker: str, ref_date: pd.Timestamp, window: int = 10) -> float | None:
    """과거 window일(당일 미포함) 중 최저 Low. 손절가 계산 기준."""
    df = load_price_cache(ticker)
    if df is None:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    past = df[df["Date"] < ref_date].tail(window)
    if past.empty:
        return None
    return float(past["Low"].min())


# ── 모델 로딩 ──────────────────────────────────────────────────────────────────

def check_model_status() -> bool:
    """모델 파일 존재 여부 확인. 없으면 경고 + False 반환."""
    cnn_loaded  = (MODELS_DIR / "saved" / "cnn_signal_c.pt").exists()
    lstm_loaded = (MODELS_DIR / "saved" / "lstm_signal_c.pt").exists()
    if not cnn_loaded or not lstm_loaded:
        logger.warning(
            "CNN %s | LSTM %s — 0.0 폴백 모드로 실행 중 (거래 비활성)",
            "✓" if cnn_loaded else "✗",
            "✓" if lstm_loaded else "✗",
        )
        return False
    return True


def _load_arch_params() -> dict:
    """models/saved/arch_params.json에 저장된 아키텍처 파라미터 로드."""
    p = MODELS_DIR / "saved" / "arch_params.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _load_models() -> tuple[CNN1DModel | None, LSTMModel | None]:
    models_ok = check_model_status()
    arch = _load_arch_params()
    cnn_model  = load_cnn(arch.get("cnn"))   if models_ok else None
    lstm_model = load_lstm(arch.get("lstm")) if models_ok else None
    return cnn_model, lstm_model


# ── 신호 계산 (100일 윈도우, 이진 라벨 모드) ─────────────────────────────────

def compute_signals(
    tickers: list[str],
    ref_date: pd.Timestamp,
    alpha: float = 0.5,
    cnn_model: CNN1DModel | None = None,
    lstm_model: LSTMModel | None = None,
) -> pd.DataFrame:
    """
    전 종목 CNN+LSTM 병렬 신호 계산.

    100일 윈도우 (cnn_100d / lstm_100d 학습 기준).
    이진 분류 모드: signal_score_bin() 사용 (P(Buy) - 신뢰도 가중).
    항상 ref_date "당일 미포함" 데이터만 사용 — 08:00/15:19 어느 시점에 불러도
    전일 종가까지만 반영되므로 장중 실시간 조회와는 무관하게 안전하다.

    Returns
    -------
    DataFrame: ticker, cnn_score, lstm_score, final_score
    """
    rows = []
    for ticker in tickers:
        price_df = load_price_cache(ticker)
        if price_df is None:
            rows.append({"ticker": ticker, "cnn_score": 0.0, "lstm_score": 0.0, "final_score": 0.0})
            continue

        price_df["Date"] = pd.to_datetime(price_df["Date"])
        # 당일 미포함 (룩어헤드 방지)
        past = price_df[price_df["Date"] < ref_date].tail(WINDOW_100D + 20)

        cnn_score  = 0.0
        lstm_score = 0.0

        # ── CNN 신호 (원시 OHLCV 100일) ──────────────────────────────────────
        if cnn_model is not None:
            try:
                past_w = past.tail(WINDOW_100D)
                if len(past_w) >= WINDOW_100D:
                    ref_close = float(past_w.iloc[-1]["Close"])
                    if ref_close > 0:
                        o = past_w["Open"].values  / ref_close
                        h = past_w["High"].values  / ref_close
                        l = past_w["Low"].values   / ref_close
                        c = past_w["Close"].values / ref_close
                        v_max = float(past_w["Volume"].max())
                        v = past_w["Volume"].values / (v_max + 1e-9)
                        x_cnn = torch.tensor(
                            np.stack([o, h, l, c, v])[np.newaxis], dtype=torch.float32
                        )
                        with torch.no_grad():
                            sig, _ = cnn_model.signal_score_bin(x_cnn)
                        cnn_score = float(sig[0])
            except Exception as e:
                logger.debug("CNN 신호 오류 %s: %s", ticker, e)

        # ── LSTM 신호 (기술 지표 100일) ───────────────────────────────────────
        if lstm_model is not None:
            try:
                tech_df   = compute_technical_features(past)
                past_tech = tech_df.tail(WINDOW_100D)
                if len(past_tech) >= WINDOW_100D:
                    x_lstm = torch.tensor(
                        past_tech.values[np.newaxis].astype(np.float32), dtype=torch.float32
                    )
                    with torch.no_grad():
                        sig, _ = lstm_model.signal_score_bin(x_lstm)
                    lstm_score = float(sig[0])
            except Exception as e:
                logger.debug("LSTM 신호 오류 %s: %s", ticker, e)

        final_score = alpha * cnn_score + (1.0 - alpha) * lstm_score
        rows.append({
            "ticker":      ticker,
            "cnn_score":   cnn_score,
            "lstm_score":  lstm_score,
            "final_score": final_score,
        })

    return pd.DataFrame(rows)


# ── C-setting 포지션 사이징 ───────────────────────────────────────────────────

def _position_pct_c(final_score: float) -> float:
    """2-tier C-setting: 0.075 이상→11%, 0.05 이상→5%, 미만→0%."""
    if final_score >= SUPER_BUY_THRESHOLD:
        return SUPER_BUY_PCT
    if final_score >= BUY_THRESHOLD:
        return BUY_POSITION_PCT
    return 0.0


def _calc_shares_c(portfolio_val: float, pct: float, entry_price: float) -> int:
    """
    목표 비중 pct 기준 주수 계산.
    - 1주 가격 > 포트폴리오 40% → 건너뜀 (0 반환)
    - shares=0으로 계산되어도 40% 이하이면 1주 진입

    entry_price는 실제 주문가(매수상한가=현재가+N틱 등) 기준으로 넘겨야 한다.
    남는 예산은 강제로 소진하지 않고 현금으로 남긴다.
    """
    if entry_price <= 0 or pct <= 0:
        return 0
    one_cost = entry_price * (1 + _COMM + _SLIP)
    if one_cost > portfolio_val * 0.40:
        return 0
    budget = portfolio_val * pct
    shares = int(budget // one_cost)
    return max(shares, 1)


def _portfolio_value(cash: float, holdings: dict, ref_date: pd.Timestamp) -> float:
    """
    포지션 사이징 기준값 = 현금 + 보유종목 매수원가 합 (시세 평가액 아님).

    안 판 종목의 평가손익(오르든 내리든)은 다음 매수 비중에 영향을 주지 않는다.
    실제로 팔아서 손익이 실현되어야만(현금 증감으로) 이 값에 반영된다 —
    "확정되지 않은 돈"으로 다음 베팅 크기를 부풀리거나 줄이지 않기 위함.
    """
    return cash + sum(
        h.get("shares", 0) * h.get("entry_price", 0)
        for t, h in holdings.items()
    )


# ── 키움 API 주문 (레거시 — 더 이상 호출되지 않음, 참고용 보존) ────────────────

def _kiwoom_order(action: str, ticker: str, shares: int, price: int) -> bool:
    try:
        from pykiwoom.kiwoom import Kiwoom
        kw = Kiwoom()
        kw.CommConnect()
        account    = os.getenv("KIWOOM_ACCOUNT", KIWOOM_CONFIG.get("account", ""))
        order_type = 1 if action == "buy" else 2
        result = kw.SendOrder(
            "order", "0101", account, order_type,
            ticker, shares, price, "00", "",
        )
        logger.info("키움 %s 주문: %s %d주 @%d — 결과: %s", action, ticker, shares, price, result)
        return result == 0
    except ImportError:
        logger.warning("pykiwoom 미설치 — 실제 주문 전송 불가 (페이퍼트레이딩 모드)")
        return False
    except Exception as e:
        logger.error("키움 주문 오류 %s: %s", ticker, e)
        return False


# ── 실제 주문 전송 (현재 브로커: 한국투자증권) ────────────────────────────────
# 모든 호출부가 send_order_fill/_submit_order를 직접 사용하도록 정리됨
# (익절처럼 "체결될 때까지 걸어두는" 주문과, 손절/청산처럼 "즉시 체결 확인 후
# 부분체결도 장부에 반영해야 하는" 주문의 성격이 달라 공용 bool 래퍼로는
# 둘 다 정확히 처리할 수 없었음).


# ── 일일 리포트 ───────────────────────────────────────────────────────────────

# 같은 날 여러 단계(진입/손절/타이머)가 리포트를 나눠 쓰다 보니 예전에는 호출할 때마다
# 파일 끝에 새 블록을 이어붙여서, 하루치 리포트 안에 실질적으로 똑같은 내용이
# "---"로 구분된 채 여러 번 중복되어 쌓였다. "하루에 리포트 하나만" 남기기 위해
# 그날 발생한 주문을 누적해서 매번 파일 전체를 새로 덮어쓴다.
#
# 이 누적 기록은 메모리가 아니라 파일(디스크)에 저장한다 — 코드 배포로 서비스를
# 재시작하면 메모리는 초기화되지만, 그러면 재시작 전에 실제로 체결된 주문이
# 리포트에서 사라져버린다(실제 거래는 정상, 리포트 표시만 유실).
_DAILY_ORDERS_PATH = BASE_DIR / "data" / "daily_orders_cache.json"


def _load_daily_orders(date_key: str) -> list[dict]:
    if not _DAILY_ORDERS_PATH.exists():
        return []
    try:
        cache = json.loads(_DAILY_ORDERS_PATH.read_text(encoding="utf-8"))
        return cache.get("orders", []) if cache.get("date") == date_key else []
    except Exception:
        return []


def _save_daily_orders(date_key: str, orders: list[dict]) -> None:
    _DAILY_ORDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DAILY_ORDERS_PATH.write_text(
        json.dumps({"date": date_key, "orders": orders}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _current_market_value(holdings: dict) -> tuple[float, list[str]]:
    """
    보유종목 평가액 합계 (15:31 장마감 확정 리포트 전용).
    이 시점은 실시간 매매 판단이 아니라 이미 끝난 하루를 "확정"하는 것이라 급할 이유가 없으므로,
    재시도를 평소(1분 감시 루프의 retries=2)보다 늘려서 웬만하면 실제 시세를 받아온다.
    그래도 실패하면 완전히 다른 API 경로인 분봉 마지막 봉 종가로 대체하고,
    그것도 실패해야 최후의 수단으로 매수원가를 쓴다 — 매수원가는 종목이 진입가 대비
    많이 움직였을수록 오차가 커지고, 조회 실패가 넓게 겹치면 리포트 수익률이
    실제와 무관하게 0%에 가깝게 나오는 위험한 실패 모드라 최우선 대체값으로 쓰지 않는다.
    """
    total = 0.0
    failed: list[str] = []
    for ticker, h in holdings.items():
        shares = h.get("shares", 0)

        quote = get_current_price(ticker, retries=5)
        if quote and quote.get("current", 0) > 0:
            total += shares * quote["current"]
            continue

        candles = get_minute_candles(ticker, count=1)
        if candles is not None and len(candles) > 0 and float(candles.iloc[0].get("close", 0)) > 0:
            total += shares * float(candles.iloc[0]["close"])
            continue

        total += shares * h.get("entry_price", 0)
        failed.append(ticker)
    return total, failed


def _write_report(
    ref_date: pd.Timestamp,
    orders: list[dict],
    portfolio_state: dict,
    signals: pd.DataFrame | None = None,
) -> None:
    fname = REPORTS_DIR / f"report_{ref_date.strftime('%Y%m%d')}.md"
    cash     = portfolio_state.get("cash", 0)
    holdings = portfolio_state.get("holdings", {})

    date_key = ref_date.strftime("%Y-%m-%d")
    all_orders = _load_daily_orders(date_key)
    all_orders.extend(orders)
    _save_daily_orders(date_key, all_orders)

    mkt_value, failed_quotes = _current_market_value(holdings)
    total_value = cash + mkt_value

    lines = [
        f"# 일일 매매 리포트 — {date_key}",
        "",
        f"## 현금: {cash:,.0f}원",
        f"## 보유종목 평가액: {mkt_value:,.0f}원",
        f"## 총 평가금액: {total_value:,.0f}원",
        f"## 보유 종목: {len(holdings)}개",
        "",
        "## 오늘 주문",
    ]
    if all_orders:
        for o in all_orders:
            lines.append(
                f"- [{o['action'].upper()}] {o['ticker']} {o['shares']}주 @{o['price']:,}원 ({o['reason']})"
            )
    else:
        lines.append("- (없음)")

    if signals is not None and not signals.empty:
        lines += ["", "## 신호 현황 (상위 10)"]
        top = signals.nlargest(10, "final_score")
        for _, r in top.iterrows():
            lines.append(
                f"- {r['ticker']}: CNN={r['cnn_score']:.3f}, LSTM={r['lstm_score']:.3f}, Final={r['final_score']:.3f}"
            )

    lines += ["", "## 보유 포지션"]
    for tk, h in holdings.items():
        entry = h["entry_price"]
        tp    = h.get("take_profit_price", 0)
        sl    = h.get("stop_loss_price", 0)
        tp_pct = (tp / entry - 1) * 100 if entry else 0.0
        sl_pct = (sl / entry - 1) * 100 if entry else 0.0
        lines.append(
            f"- {tk}: {h['shares']}주, 진입가={entry:,}원, "
            f"보유일={h.get('hold_days',0)}, TP={tp:,}원({tp_pct:+.1f}%), SL={sl:,}원({sl_pct:+.1f}%)"
        )

    if failed_quotes:
        lines += ["", f"(참고: {', '.join(failed_quotes)} 실시간 시세 조회 실패 — 매수원가로 대체 계산됨)"]

    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("리포트 저장: %s", fname)


def _email_current_report(ref_date: pd.Timestamp, subject_prefix: str) -> None:
    """그날 누적된 리포트 파일 전체를 이메일로 발송."""
    from utils.email_notifier import send_report_email

    fname = REPORTS_DIR / f"report_{ref_date.strftime('%Y%m%d')}.md"
    if not fname.exists():
        return
    body = fname.read_text(encoding="utf-8")
    subject = f"[퀀트봇] {subject_prefix} — {ref_date.strftime('%Y-%m-%d')}"
    send_report_email(subject, body)


# ══════════════════════════════════════════════════════════════════════════
# Phase 1 — 08:00  전일 종가 기준 매수 후보 계산
# ══════════════════════════════════════════════════════════════════════════

def run_daily_signal(ref_date: str | None = None, alpha: float = 0.5) -> dict:
    """
    매일 08:00 실행. 실시간 시세와 무관 — 전일 종가까지의 데이터만 사용한다.
    Agent B 풀 + 보유 종목에 대해 신호를 계산하고, 신규 매수 후보를
    pending_buys에 등록한다. holdings/cash는 건드리지 않는다
    (실제 체결은 09:01 run_market_open_entry에서 이루어짐).
    """
    if ref_date is None:
        ref_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    ref = pd.Timestamp(ref_date)

    cnn_model, lstm_model = _load_models()

    tickers = _get_top30_pool(ref_date=ref)
    if not tickers:
        logger.warning("Agent B 풀 없음 — factor_dataset_B.parquet / xgb_agent_B_live.pkl 확인")
        return {"status": "no_pool"}

    portfolio    = _load_portfolio()
    holdings     = portfolio.get("holdings", {})
    pending_buys = portfolio.get("pending_buys", {})

    all_tickers = list(dict.fromkeys(tickers + list(holdings.keys())))

    logger.info("Agent C 신호 계산: %d종목 (기준일 %s)", len(all_tickers), ref_date)
    signals = compute_signals(all_tickers, ref, alpha, cnn_model, lstm_model)

    for _, sig_row in signals.iterrows():
        ticker = sig_row["ticker"]
        fs     = float(sig_row["final_score"])
        if fs >= BUY_THRESHOLD and ticker not in holdings and ticker not in pending_buys:
            pending_buys[ticker] = fs

    portfolio["pending_buys"] = pending_buys
    _save_portfolio(portfolio)

    # 다음 단계(09:01, 15:19)에서 재사용할 수 있도록 오늘 신호를 캐시해둔다.
    sig_cache_path = BASE_DIR / "data" / "today_signals.parquet"
    signals["date"] = ref_date
    signals.to_parquet(sig_cache_path, index=False)

    logger.info("매수 후보 갱신 완료: pending_buys=%d종목", len(pending_buys))
    return {"status": "ok", "date": ref_date, "n_pending": len(pending_buys), "signals": signals}


# ══════════════════════════════════════════════════════════════════════════
# Phase 2 — 09:01  실시간 시가 확인 + 매수 실행 + 익절 주문 재등록
# ══════════════════════════════════════════════════════════════════════════

def _sync_with_broker(portfolio: dict, paper_trading: bool) -> dict:
    """
    portfolio_state.json은 시스템이 자체 계산으로 관리하는 장부일 뿐,
    실제 KIS 계좌 잔고를 조회해서 확인한 적이 없었다. 페이퍼트레이딩이
    아닐 때만 실제 잔고를 조회해 현금을 동기화하고, 보유종목이 어긋나면
    (실제 계좌엔 있는데 우리 장부엔 없는 경우 등) 자동으로 고치지 않고
    로그로 명확히 경고한다 — entry_price/TP/SL 같은 정보가 없어 임의로
    합칠 수 없기 때문이다.
    """
    if paper_trading:
        return portfolio

    real = get_account_balance()
    if real is None:
        logger.warning("실제 계좌 잔고 조회 실패 — 기존 장부(portfolio_state.json)로 계속 진행")
        return portfolio

    tracked_cash = float(portfolio.get("cash", INITIAL_CAPITAL))
    if abs(real["cash"] - tracked_cash) > 1:
        logger.warning(
            "현금 잔고 불일치 감지 — 장부=%.0f원, 실제 계좌=%.0f원. 실제 계좌 기준으로 동기화",
            tracked_cash, real["cash"],
        )
        portfolio["cash"] = real["cash"]

    tracked_tickers = set(portfolio.get("holdings", {}).keys())
    real_tickers    = set(real["holdings"].keys())
    only_in_broker  = real_tickers - tracked_tickers
    only_in_tracked = tracked_tickers - real_tickers
    if only_in_broker:
        logger.warning(
            "실제 계좌에는 있는데 장부엔 없는 종목 발견(수동 거래 등으로 추정) — 확인 필요: %s",
            sorted(only_in_broker),
        )
    if only_in_tracked:
        logger.warning(
            "장부엔 있는데 실제 계좌엔 없는 종목 발견 — 확인 필요: %s",
            sorted(only_in_tracked),
        )

    return portfolio


def run_market_open_entry(ref_date: str | None = None, paper_trading: bool = True) -> dict:
    """
    매일 09:01 실행 (KRX 본장 09:00 개장 직후).

    ① 보유 종목: 그날의 익절 지정가 매도 주문을 재등록한다
       (전일 미체결 주문은 장마감 시 자동취소되므로 매일 다시 걸어야 함).
    ② pending_buys: 실시간 시가를 조회해 갭 필터(ATR×1.5) 판단 후,
       현재가+N틱 지정가로 매수한다 (예산도 이 상한가 기준으로 계산).

    실제 주문 모드에서는 시작 시 실제 계좌 잔고와 장부를 동기화한다.
    """
    if ref_date is None:
        ref_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    ref = pd.Timestamp(ref_date)

    portfolio    = _load_portfolio()
    portfolio    = _sync_with_broker(portfolio, paper_trading)
    holdings     = portfolio.get("holdings", {})
    pending_buys = portfolio.get("pending_buys", {})
    cash         = float(portfolio.get("cash", INITIAL_CAPITAL))

    orders: list[dict] = []

    # ── ① 보유 종목 익절 주문 재등록 ─────────────────────────────────────────
    # 익절은 "체결될 때까지 하루 종일 걸어두는" 지정가 주문이라 send_order_fill류의
    # 대기 후 미체결시 취소 로직을 쓰면 안 된다(정상적으로 몇 시간 뒤에나 체결될
    # 주문을 몇 초만에 취소해버리게 됨). _submit_order로 접수만 하고 odno를
    # 저장해둔 뒤 check_tp_fills가 주기적으로 체결 여부를 확인한다.
    for ticker, holding in holdings.items():
        tp_price = holding.get("take_profit_price")
        shares   = holding.get("shares", 0)
        if not tp_price or shares <= 0:
            continue
        if paper_trading:
            logger.debug("[페이퍼] %s 익절 주문 재등록 생략: %d주 @%.0f", ticker, shares, tp_price)
            continue
        from utils.kis_client import _submit_order
        submitted = _submit_order("sell", ticker, shares, int(tp_price))
        if submitted:
            holding["tp_odno"] = submitted.get("odno")
            holding["tp_order_qty"] = shares
            logger.info("%s 익절 주문 재등록: %d주 @%.0f (odno=%s)", ticker, shares, tp_price, submitted.get("odno"))
        else:
            holding["tp_odno"] = None
            holding["tp_order_qty"] = None
            logger.warning("%s 익절 주문 재등록 실패 — 다음 체크(장중 손절)로 보완됨", ticker)

    # ── ② 신규 매수 ───────────────────────────────────────────────────────────
    total_val = _portfolio_value(cash, holdings, ref)
    executed: list[str] = []

    for ticker, prev_score in list(pending_buys.items()):
        if ticker in holdings:
            executed.append(ticker)
            continue

        quote = get_current_price(ticker)
        if quote is None:
            logger.warning("%s 실시간 시세 조회 실패 — 이번 회차 스킵 (pending 유지)", ticker)
            continue

        open_price = quote["open"]
        prev_close = quote["prev_close"]
        if open_price <= 0 or prev_close <= 0:
            executed.append(ticker)  # 거래정지 등으로 시세 자체가 무의미 — 후보에서 제외
            continue

        # ATR(10) 갭 필터 — 전일까지 로컬 캐시로 계산, 실시간 시가와 비교
        atr10 = _compute_atr(ticker, ref, window=_GAP_WIN)
        if atr10 is None:
            executed.append(ticker)
            continue
        gap = open_price - prev_close
        if gap > _GAP_MULT * atr10:
            logger.info(
                "%s 갭 필터 스킵: gap=%.0f > ATR×%.1f=%.0f (다음날 재평가)",
                ticker, gap, _GAP_MULT, _GAP_MULT * atr10,
            )
            continue  # pending 유지 — 다음날 독립적으로 재평가

        # 손절가 (과거 10일 Low × (1-buffer)) — 전일까지 로컬 캐시 기준
        swing_low = _swing_low(ticker, ref, window=_SL_WIN)
        if swing_low is None:
            executed.append(ticker)
            continue
        sl_price = swing_low * (1 - _SL_BUF)

        if sl_price >= open_price:
            logger.debug("%s 진입 취소: SL(%.0f) >= 시가(%.0f)", ticker, sl_price, open_price)
            executed.append(ticker)
            continue

        tp_price = round_to_tick(open_price + (open_price - sl_price) * _TP_RR)

        # 매수상한가 = 현재가(시가) + N틱 — 이 가격 기준으로 예산도 계산
        buy_ceiling = price_plus_ticks(open_price, _BUY_TICKS)

        pct    = _position_pct_c(float(prev_score))
        shares = _calc_shares_c(total_val, pct, buy_ceiling)
        if shares <= 0:
            executed.append(ticker)
            continue

        cost = buy_ceiling * shares * (1 + _COMM + _SLIP)
        if cost > cash:
            logger.debug("%s 현금 부족 스킵: 필요=%.0f, 보유=%.0f", ticker, cost, cash)
            continue  # 현금 부족은 일시적일 수 있으니 pending 유지

        entry_price_used = open_price  # 계산 기준가 (전량 체결 시 기존 방식 유지)

        if not paper_trading:
            from utils.kis_client import send_order_fill
            fill = send_order_fill("buy", ticker, shares, buy_ceiling)
            filled_qty = fill["filled_qty"]

            if filled_qty <= 0:
                logger.warning("%s 매수 전량 미체결 — 1분 뒤 남은 %d주 재시도 등록", ticker, shares)
                portfolio.setdefault("buy_retry", {})[ticker] = {
                    "date": ref_date, "remaining": shares,
                    "take_profit_price": round(tp_price, 2), "stop_loss_price": round(sl_price, 2),
                    "final_score": round(float(prev_score), 4),
                }
                executed.append(ticker)  # pending_buys에서는 제거 — 이제 buy_retry가 담당
                continue

            if filled_qty < shares:
                logger.warning("%s 매수 부분체결 %d/%d주 — 나머지는 1분 뒤 재시도 등록", ticker, filled_qty, shares)
                portfolio.setdefault("buy_retry", {})[ticker] = {
                    "date": ref_date, "remaining": shares - filled_qty,
                    "take_profit_price": round(tp_price, 2), "stop_loss_price": round(sl_price, 2),
                    "final_score": round(float(prev_score), 4),
                }
                shares = filled_qty
                entry_price_used = fill["avg_price"] or open_price
                cost = entry_price_used * shares * (1 + _COMM + _SLIP)

        cash -= cost
        holdings[ticker] = {
            "shares":            shares,
            "entry_price":       round(entry_price_used, 2),   # 계산 기준가 (실제 체결가는 상한가 이하)
            "order_price":       buy_ceiling,             # 실제 제출한 주문가
            "entry_date":        ref_date,
            "take_profit_price": round(tp_price, 2),
            "stop_loss_price":   round(sl_price, 2),
            "hold_days":         0,
            "sell_signal_days":  0,
            "final_score":       round(float(prev_score), 4),
        }

        logger.info(
            "매수: %s %d주 @%d원 (시가=%.0f, TP=%.0f, SL=%.0f, pct=%.0f%%, score=%.3f)",
            ticker, shares, buy_ceiling, open_price, tp_price, sl_price, pct * 100, float(prev_score),
        )

        # 매수 당일부터 익절 주문이 살아있도록 즉시 등록
        # (다음날 아침 재등록 로직은 그대로 유지 — 당일 미체결분은 장마감 시 자동취소되므로 계속 필요)
        # 매수 직후 계좌 반영까지 시차가 있어(모의투자 서버 기준) "잔고내역이 없습니다" 오류가
        # 날 수 있음 — 매수 API 재시도로 지연될 상황까지 감안해 10초 대기 후 등록
        if not paper_trading:
            time.sleep(10)
            from utils.kis_client import _submit_order
            tp_submitted = _submit_order("sell", ticker, shares, int(round(tp_price)))
            if tp_submitted:
                holdings[ticker]["tp_odno"] = tp_submitted.get("odno")
                holdings[ticker]["tp_order_qty"] = shares
                logger.info("%s 당일 익절 주문 등록: %d주 @%.0f (odno=%s)", ticker, shares, tp_price, tp_submitted.get("odno"))
            else:
                holdings[ticker]["tp_odno"] = None
                holdings[ticker]["tp_order_qty"] = None
                logger.warning("%s 당일 익절 주문 등록 실패 — 다음날 아침 재등록으로 보완됨", ticker)

        orders.append({
            "action": "buy", "ticker": ticker, "shares": shares,
            "price": buy_ceiling,
            "reason": f"score={float(prev_score):.3f} pct={pct*100:.0f}%",
        })
        executed.append(ticker)

    for t in executed:
        pending_buys.pop(t, None)

    portfolio["holdings"]     = holdings
    portfolio["pending_buys"] = pending_buys
    portfolio["cash"]         = round(cash, 2)
    _save_portfolio(portfolio)

    _write_report(ref, orders, portfolio)
    # 이메일은 여기서 바로 안 보낸다 — 09:01에 전량 미체결된 종목은 check_buy_retry가
    # 1분마다 재시도해서 몇 분 뒤에나 체결되는데, 여기서 바로 보내면 그 재시도 체결분이
    # 리포트에서 통째로 빠진다(2026-07-13 첫 실전 거래일에 실제로 확인됨 — 9종목 중
    # 7종목이 재시도로 체결됨). 재시도가 어느 정도 끝난 09:10에 main.py의
    # job_morning_report가 그때까지 누적된 리포트 파일을 이메일로 보낸다.

    logger.info(
        "09:01 진입 완료: 매수=%d, 현금=%.0f원, 보유=%d종목",
        sum(1 for o in orders if o["action"] == "buy"), cash, len(holdings),
    )
    return {"status": "ok", "date": ref_date, "orders": orders, "cash": cash, "n_held": len(holdings)}


# ══════════════════════════════════════════════════════════════════════════
# Phase 3 — 09:00~15:19  1분마다 장중 손절 감시 (15:20~15:30 동시호가 구간 제외)
# ══════════════════════════════════════════════════════════════════════════

def check_intraday_stop_loss(paper_trading: bool = True) -> dict:
    """
    1분마다 실행. 보유 종목의 분봉을 조회해 "마지막 체크 이후 구간"에
    손절선을 스친 적이 있는지 확인한다 (스냅샷이 아닌 구간 저가 기준이라
    체크 사이 짧은 급락도 놓치지 않는다).

    한투는 역지정가(스톱) 주문을 지원하지 않아 이 방식으로 대체한다.
    """
    portfolio = _load_portfolio()
    holdings  = portfolio.get("holdings", {})
    if not holdings:
        return {"status": "ok", "n_exit": 0}

    cash = float(portfolio.get("cash", INITIAL_CAPITAL))
    to_exit: list[str] = []
    orders: list[dict] = []

    for ticker, holding in list(holdings.items()):
        sl_price = holding.get("stop_loss_price")
        if not sl_price:
            continue

        candles = get_minute_candles(ticker, count=30)
        if candles is None or candles.empty:
            logger.debug("%s 분봉 조회 실패 — 이번 회차 스킵", ticker)
            continue

        recent_low = float(candles["low"].min())
        if recent_low > sl_price:
            continue

        quote = get_current_price(ticker)
        if quote is None or quote["current"] <= 0:
            logger.warning("%s 손절선 감지됐으나 현재가 조회 실패 — 다음 회차 재시도", ticker)
            continue

        sell_floor = price_minus_ticks(quote["current"], _SELL_TICKS)
        shares = holding.get("shares", 0)

        if not paper_trading:
            # 걸어둔 익절 주문이 남아있으면 그 수량만큼 매도가능 수량이 이미
            # 잠겨있어 손절 매도가 거부될 수 있다(2026-07-09 포스코DX 건에서
            # 실제로 확인됨: 손절 시도가 "잔고내역이 없습니다"로 계속 실패) —
            # 먼저 취소해서 수량을 풀어준다.
            tp_odno = holding.get("tp_odno")
            if tp_odno:
                from utils.kis_client import get_order_fill_status, cancel_order
                status = get_order_fill_status(ticker, tp_odno)
                if status:
                    cancel_order(
                        status.get("ord_gno_brno"), tp_odno,
                        holding.get("tp_order_qty", shares), int(holding.get("take_profit_price", sell_floor)),
                    )
                holding["tp_odno"] = None
                holding["tp_order_qty"] = None

            from utils.kis_client import send_order_fill
            fill = send_order_fill("sell", ticker, shares, sell_floor)
            filled_qty = fill["filled_qty"]
            if filled_qty <= 0:
                logger.warning("%s 손절 주문 전량 미체결 — 다음 회차 재시도", ticker)
                continue

            avg_price = fill["avg_price"] or sell_floor
            cash += avg_price * filled_qty * (1 - _COMM - _TAX - _SLIP)

            if filled_qty < shares:
                holding["shares"] = shares - filled_qty
                logger.warning(
                    "손절 부분체결: %s %d/%d주 @%.0f원 (SL=%.0f) — 잔여 %d주는 다음 회차 재시도",
                    ticker, filled_qty, shares, avg_price, sl_price, shares - filled_qty,
                )
                orders.append({
                    "action": "sell", "ticker": ticker, "shares": filled_qty,
                    "price": round(avg_price), "reason": "stop_loss",
                })
                continue

            logger.info("손절 체결: %s %d주 @%.0f원 (SL=%.0f, 구간저가=%.0f)", ticker, shares, avg_price, sl_price, recent_low)
            orders.append({
                "action": "sell", "ticker": ticker, "shares": shares,
                "price": round(avg_price), "reason": "stop_loss",
            })
            to_exit.append(ticker)
            continue

        cash += sell_floor * shares * (1 - _COMM - _TAX - _SLIP)
        logger.info("손절 체결: %s %d주 @%d원 (SL=%.0f, 구간저가=%.0f)", ticker, shares, sell_floor, sl_price, recent_low)
        orders.append({
            "action": "sell", "ticker": ticker, "shares": shares,
            "price": sell_floor, "reason": "stop_loss",
        })
        to_exit.append(ticker)

    if not orders:
        return {"status": "ok", "n_exit": 0}

    # 부분체결만 있고 완전 청산은 없는 회차도 반드시 저장해야 한다 — 안 그러면
    # holding["shares"]가 줄어든 게(예: 71→70) 파일에 반영 안 된 채로 다음 회차가
    # 여전히 옛 수량(71)으로 매도를 재시도해 "잔고내역이 없습니다" 오류가 반복된다
    # (2026-07-10 439260 건에서 실제로 확인됨: to_exit만 보고 조기 return 하던 버그).
    for t in to_exit:
        del holdings[t]

    portfolio["holdings"] = holdings
    portfolio["cash"]     = round(cash, 2)
    _save_portfolio(portfolio)

    ref = pd.Timestamp.today().normalize()
    _write_report(ref, orders, portfolio)

    return {"status": "ok", "n_exit": len(to_exit), "orders": orders}


def check_buy_retry(paper_trading: bool = True) -> dict:
    """
    1분마다 실행 — 매수 주문이 미체결/부분체결로 남은 잔여 수량을,
    그 시점 현재가 + (매수 틱오프셋+2)틱으로 재시도한다.

    09:01에 처음 매수를 시도했을 때 전량 체결이 안 되면, 원래 수량 전체를
    "다음 거래일"에 다시 사는 대신 남은 수량만 1분 뒤 더 높은 가격으로
    재시도해서 당일 안에 체결시키는 걸 우선한다.
    """
    if paper_trading:
        return {"status": "ok", "n_retry": 0}

    portfolio = _load_portfolio()
    retry_map: dict = portfolio.get("buy_retry", {})
    if not retry_map:
        return {"status": "ok", "n_retry": 0}

    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    holdings  = portfolio.get("holdings", {})
    cash      = float(portfolio.get("cash", INITIAL_CAPITAL))
    orders: list[dict] = []

    from utils.kis_client import send_order_fill

    done: list[str] = []
    for ticker, info in list(retry_map.items()):
        if info.get("date") != today_str:
            logger.warning("%s 매수 재시도 정보가 오늘 것이 아님 — 폐기", ticker)
            done.append(ticker)
            continue

        remaining = info["remaining"]
        quote = get_current_price(ticker)
        if quote is None or quote["current"] <= 0:
            continue

        buy_price = price_plus_ticks(quote["current"], _BUY_TICKS + 2)
        cost = buy_price * remaining * (1 + _COMM + _SLIP)
        if cost > cash:
            logger.debug("%s 매수 재시도 현금 부족 — 다음 회차 재시도", ticker)
            continue

        fill = send_order_fill("buy", ticker, remaining, buy_price)
        filled_qty = fill["filled_qty"]
        if filled_qty <= 0:
            continue  # 다음 1분 회차에 다시 시도

        avg_price   = fill["avg_price"] or buy_price
        cost_filled = avg_price * filled_qty * (1 + _COMM + _SLIP)
        cash -= cost_filled

        if ticker in holdings:
            old_shares = holdings[ticker]["shares"]
            new_shares = old_shares + filled_qty
            new_entry  = (holdings[ticker]["entry_price"] * old_shares + avg_price * filled_qty) / new_shares
            holdings[ticker]["shares"]      = new_shares
            holdings[ticker]["entry_price"] = round(new_entry, 2)

            # 익절가는 진입가 기준으로 산출되는 값이라(entry + (entry-SL)*RR),
            # 평균 진입가가 바뀌면 반드시 같이 재계산해야 한다. 예전 값을 그대로 쓰면
            # 재시도 사이 가격이 오른 만큼 새 평균 진입가가 옛 익절가보다 높아질 수 있는데,
            # 그 상태로 "옛 익절가"를 그대로 재등록하면 사자마자 원가 밑으로 파는 주문이
            # 걸려서 즉시 체결돼버린다 — "익절"이라는 이름표만 붙은 손실 거래
            # (2026-07-14 022100/포스코DX 건에서 실제로 확인: 체결가 18,768원인데
            # 옛 익절가 18,720원이 그대로 재등록·즉시 체결됨).
            sl_price = holdings[ticker].get("stop_loss_price", 0)
            new_tp   = round_to_tick(new_entry + (new_entry - sl_price) * _TP_RR) if sl_price else None

            old_tp_odno = holdings[ticker].get("tp_odno")
            if old_tp_odno:
                from utils.kis_client import get_order_fill_status, cancel_order, _submit_order
                status = get_order_fill_status(ticker, old_tp_odno)
                if status:
                    cancel_order(
                        status.get("ord_gno_brno"), old_tp_odno,
                        holdings[ticker].get("tp_order_qty", old_shares),
                        int(holdings[ticker].get("take_profit_price", new_tp or 0)),
                    )
            if new_tp:
                holdings[ticker]["take_profit_price"] = new_tp
                from utils.kis_client import _submit_order
                resubmitted = _submit_order("sell", ticker, new_shares, int(new_tp))
                if resubmitted:
                    holdings[ticker]["tp_odno"]      = resubmitted.get("odno")
                    holdings[ticker]["tp_order_qty"] = new_shares
                else:
                    holdings[ticker]["tp_odno"]      = None
                    holdings[ticker]["tp_order_qty"] = None
            else:
                holdings[ticker]["tp_odno"]      = None
                holdings[ticker]["tp_order_qty"] = None
        else:
            sl_price = info["stop_loss_price"]

            if sl_price >= avg_price:
                # 역전 방지: run_market_open_entry의 "SL>=시가면 진입취소"와 같은 취지.
                # 여기는 이미 체결된 뒤라 주문을 취소할 수 없으니, 즉시 반대매매로 되팔아
                # 손절 구조가 깨진 포지션을 만들지 않는다 (재시도 사이 가격이 떨어져
                # 체결가가 손절선 이하로 내려간 극단적인 경우에 대한 안전장치).
                quote = get_current_price(ticker)
                unwind_price = (
                    price_minus_ticks(quote["current"], _SELL_TICKS)
                    if quote and quote.get("current", 0) > 0 else int(avg_price)
                )
                unwind = send_order_fill("sell", ticker, filled_qty, unwind_price)
                unwind_qty = unwind["filled_qty"]
                if unwind_qty > 0:
                    unwind_avg = unwind["avg_price"] or unwind_price
                    cash += unwind_avg * unwind_qty * (1 - _COMM - _TAX - _SLIP)
                    logger.warning(
                        "%s 매수 재시도 체결가(%.0f)가 손절가(%.0f) 이하로 역전 — 즉시 반대매매 청산 %d주 @%.0f",
                        ticker, avg_price, sl_price, unwind_qty, unwind_avg,
                    )
                    orders.append({
                        "action": "sell", "ticker": ticker, "shares": unwind_qty,
                        "price": round(unwind_avg), "reason": "entry_sl_inversion_unwind",
                    })
                if unwind_qty < filled_qty:
                    # 반대매매마저 부분체결이면 남은 물량은 포지션으로 남되 익절 주문은
                    # 안 건다(이미 손절선 이하라 익절가 계산 자체가 무의미) — 다음
                    # 1분 손절 감시가 바로 청산을 시도한다.
                    holdings[ticker] = {
                        "shares": filled_qty - unwind_qty, "entry_price": round(avg_price, 2),
                        "order_price": buy_price, "entry_date": today_str,
                        "take_profit_price": avg_price, "stop_loss_price": sl_price,
                        "hold_days": 0, "sell_signal_days": 0, "final_score": info["final_score"],
                        "tp_odno": None, "tp_order_qty": None,
                    }
            else:
                # 익절가는 09:01 첫 시도 때의 가정 가격(info["take_profit_price"])이 아니라
                # 이 재시도의 실제 체결가(avg_price) 기준으로 다시 계산해야 한다
                # (재시도 사이 가격이 올라 실제 체결가가 옛 익절가를 넘어서면, 그 옛 값을
                # 그대로 등록하는 순간 원가보다 싸게 파는 주문이 즉시 체결된다 —
                # 2026-07-14 022100/포스코DX 건에서 실제로 확인).
                new_tp = round_to_tick(avg_price + (avg_price - sl_price) * _TP_RR)

                holdings[ticker] = {
                    "shares":            filled_qty,
                    "entry_price":       round(avg_price, 2),
                    "order_price":       buy_price,
                    "entry_date":        today_str,
                    "take_profit_price": new_tp,
                    "stop_loss_price":   sl_price,
                    "hold_days":         0,
                    "sell_signal_days":  0,
                    "final_score":       info["final_score"],
                }
                # 09:01에 전량 미체결이라 이 재시도로 처음 보유가 생긴 신규 종목은
                # run_market_open_entry의 매수 직후 익절등록 단계를 거치지 않고 지나가서
                # 익절 주문이 아예 안 걸리는 버그가 있었다(2026-07-13 첫 실전 거래일 —
                # 103140/035720/004020/064350/068270/112610/064400 7종목에서 실제 확인).
                # 여기서 매수 성공 직후 바로 익절 주문을 등록해야 한다.
                from utils.kis_client import _submit_order
                tp_submitted = _submit_order("sell", ticker, filled_qty, int(new_tp))
                if tp_submitted:
                    holdings[ticker]["tp_odno"]      = tp_submitted.get("odno")
                    holdings[ticker]["tp_order_qty"] = filled_qty
                    logger.info("%s 매수 재시도 체결 직후 익절 주문 등록: %d주 @%.0f (odno=%s)",
                                ticker, filled_qty, new_tp, tp_submitted.get("odno"))
                else:
                    holdings[ticker]["tp_odno"]      = None
                    holdings[ticker]["tp_order_qty"] = None
                    logger.warning("%s 매수 재시도 체결 직후 익절 주문 등록 실패 — 다음 손절 감시로 보완됨", ticker)

        info["remaining"] -= filled_qty
        logger.info("%s 매수 재시도 체결: %d주 @%.0f (잔여 %d주)", ticker, filled_qty, avg_price, info["remaining"])
        orders.append({
            "action": "buy", "ticker": ticker, "shares": filled_qty,
            "price": round(avg_price), "reason": "매수 재시도 체결",
        })
        if info["remaining"] <= 0:
            done.append(ticker)

    for t in done:
        del retry_map[t]

    portfolio["holdings"]  = holdings
    portfolio["cash"]      = round(cash, 2)
    portfolio["buy_retry"] = retry_map
    _save_portfolio(portfolio)

    # 재시도 체결분도 "오늘 주문" 기록에 남겨야 리포트가 실제 계좌와 어긋나지 않는다
    # (지금까지는 장부만 갱신되고 리포트의 주문 목록엔 안 남아 사장님이 발견한 불일치의 원인이었음).
    if orders:
        _write_report(pd.Timestamp(today_str), orders, portfolio)
    return {"status": "ok", "n_retry": len(retry_map)}


def check_tp_fills(paper_trading: bool = True) -> dict:
    """
    1분마다 실행 — 걸어둔 익절 지정가 주문(holdings[ticker]["tp_odno"])의
    누적 체결 수량을 조회해, 시스템이 모르는 새 부분/전량 체결이 있었는지 확인한다.

    익절 주문은 "하루 종일 걸어두고 기다리는" 지정가 주문이라 체결까지 몇 시간이
    걸리는 게 정상이다. 그래서 접수 후 짧게 기다렸다 미체결이면 취소하는
    send_order_fill 방식을 쓰면 안 되고(정상 동작 중인 주문을 스스로 취소해버림),
    _submit_order로 접수만 해두고 이 함수가 주기적으로 진짜 체결량만 확인한다.

    2026-07-09 포스코DX(022100) 건: 전일 걸어둔 익절 주문이 41/72주 부분체결된 채
    시스템은 여전히 72주 보유 중으로 알고 있었음 — 체결 확인 로직이 아예 없었던 게 원인.
    """
    if paper_trading:
        return {"status": "ok", "n_filled": 0}

    portfolio = _load_portfolio()
    holdings  = portfolio.get("holdings", {})
    if not holdings:
        return {"status": "ok", "n_filled": 0}

    from utils.kis_client import get_order_fill_status

    cash = float(portfolio.get("cash", INITIAL_CAPITAL))
    orders: list[dict] = []
    to_remove: list[str] = []
    n_filled = 0

    for ticker, holding in list(holdings.items()):
        tp_odno      = holding.get("tp_odno")
        tp_order_qty = holding.get("tp_order_qty")
        if not tp_odno or not tp_order_qty:
            continue

        status = get_order_fill_status(ticker, tp_odno)
        if status is None:
            continue

        filled_qty = status["filled_qty"]
        if filled_qty <= 0:
            continue

        # tp_order_qty(주문 낼 때의 수량) - filled_qty(누적 체결) = 지금 시장에 남아있는 수량.
        # 이게 우리가 알고 있는 현재 보유 수량보다 적다면, 그 차이만큼 우리가 모르는 새
        # 체결이 있었다는 뜻이다.
        new_shares = max(0, tp_order_qty - filled_qty)
        old_shares = holding.get("shares", 0)
        if new_shares >= old_shares:
            continue

        sold      = old_shares - new_shares
        avg_price = status.get("avg_price") or holding.get("take_profit_price", 0)
        cash += avg_price * sold * (1 - _COMM - _TAX - _SLIP)
        n_filled += 1

        logger.info(
            "익절 체결 확인: %s %d주 @%.0f (주문 %d주 중 누적체결 %d주, 남은 보유 %d주)",
            ticker, sold, avg_price, tp_order_qty, filled_qty, new_shares,
        )
        orders.append({
            "action": "sell", "ticker": ticker, "shares": sold,
            "price": round(avg_price), "reason": "take_profit",
        })

        if new_shares <= 0:
            to_remove.append(ticker)
        else:
            holding["shares"] = new_shares

    if n_filled == 0:
        return {"status": "ok", "n_filled": 0}

    for t in to_remove:
        holdings.pop(t, None)

    portfolio["holdings"] = holdings
    portfolio["cash"]     = round(cash, 2)
    _save_portfolio(portfolio)

    ref = pd.Timestamp.today().normalize()
    _write_report(ref, orders, portfolio)

    return {"status": "ok", "n_filled": n_filled, "orders": orders}


# ══════════════════════════════════════════════════════════════════════════
# Phase 4 — 15:19  타이머·신호청산 (정규장 종가 단일가매매 시작 전)
# ══════════════════════════════════════════════════════════════════════════

def run_end_of_day_settlement(ref_date: str | None = None, alpha: float = 0.5, paper_trading: bool = True) -> dict:
    """
    매일 15:10 실행. "며칠 보유했나 / 며칠 연속 매도신호인가"를 판정하는 부분(Step 1/2)은
    날짜 단위 카운터라 하루 한 번만 계산해야 한다 — 여기서 그 판정과 첫 매도 시도를 한다.

    Step 1: 보유일(hold_days) +1, 10영업일 도달 시 타이머 청산
    Step 2: 4일 연속 매도신호(final_score < 0) 시 신호청산
    둘 다 현재가-N틱 지정가로 즉시 체결 시도. 15:19까지 시간 여유를 남겨서, 여기서
    미체결/부분체결된 건은 eod_exit_retry에 등록해 check_eod_exit_retry()가
    1분마다(15:10~15:19) 잔여 수량만 재시도한다 — 손절과 동일하게 당일 안에 최대한
    체결시키는 게 목표. 그래도 15:19까지 안 되면 hold_days가 계속 누적되므로
    다음 거래일 15:10에 다시 판정된다.
    """
    if ref_date is None:
        ref_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    ref = pd.Timestamp(ref_date)

    portfolio = _load_portfolio()
    holdings  = portfolio.get("holdings", {})
    if not holdings:
        return {"status": "ok", "n_exit": 0}

    cash = float(portfolio.get("cash", INITIAL_CAPITAL))

    # 보유 종목만 신호 재계산 (신호 자체는 전일 종가 기준이라 08:00과 동일 결과)
    cnn_model, lstm_model = _load_models()
    signals = compute_signals(list(holdings.keys()), ref, alpha, cnn_model, lstm_model)

    to_exit: list[tuple[str, str]] = []
    for ticker, holding in list(holdings.items()):
        # Step 1: 타이머
        hold_days = holding.get("hold_days", 0) + 1
        holding["hold_days"] = hold_days
        if hold_days >= _MAX_HOLD:
            to_exit.append((ticker, "max_hold"))
            continue

        # Step 2: 4일 연속 매도신호
        sig_row = signals[signals["ticker"] == ticker]
        fs = float(sig_row.iloc[0]["final_score"]) if not sig_row.empty else 0.0
        sell_days = holding.get("sell_signal_days", 0)
        sell_days = sell_days + 1 if fs < 0 else 0
        holding["sell_signal_days"] = sell_days
        if sell_days >= SIGNAL_THRESHOLDS["sell_consecutive"]:
            to_exit.append((ticker, "signal_exit"))

    today_str = ref.strftime("%Y-%m-%d")
    eod_exit_retry: dict = portfolio.get("eod_exit_retry", {})
    orders: list[dict] = []
    for ticker, reason in to_exit:
        shares = holdings[ticker].get("shares", 0)
        quote = get_current_price(ticker)
        if quote is None or quote["current"] <= 0:
            logger.warning("%s %s 청산 대상이나 현재가 조회 실패 — 15:19까지 1분마다 재시도", ticker, reason)
            eod_exit_retry[ticker] = {"reason": reason, "remaining": shares, "date": today_str}
            continue

        sell_floor = price_minus_ticks(quote["current"], _SELL_TICKS)

        if not paper_trading:
            tp_odno = holdings[ticker].get("tp_odno")
            if tp_odno:
                from utils.kis_client import get_order_fill_status, cancel_order
                status = get_order_fill_status(ticker, tp_odno)
                if status:
                    cancel_order(
                        status.get("ord_gno_brno"), tp_odno,
                        holdings[ticker].get("tp_order_qty", shares),
                        int(holdings[ticker].get("take_profit_price", sell_floor)),
                    )
                holdings[ticker]["tp_odno"] = None
                holdings[ticker]["tp_order_qty"] = None

            from utils.kis_client import send_order_fill
            fill = send_order_fill("sell", ticker, shares, sell_floor)
            filled_qty = fill["filled_qty"]
            if filled_qty <= 0:
                logger.warning("%s %s 청산 주문 전량 미체결 — 15:19까지 1분마다 재시도", ticker, reason)
                eod_exit_retry[ticker] = {"reason": reason, "remaining": shares, "date": today_str}
                continue

            avg_price = fill["avg_price"] or sell_floor
            cash += avg_price * filled_qty * (1 - _COMM - _TAX - _SLIP)

            if filled_qty < shares:
                holdings[ticker]["shares"] = shares - filled_qty
                logger.warning(
                    "청산(%s) 부분체결: %s %d/%d주 @%.0f원 — 잔여 %d주는 15:19까지 1분마다 재시도",
                    reason, ticker, filled_qty, shares, avg_price, shares - filled_qty,
                )
                eod_exit_retry[ticker] = {"reason": reason, "remaining": shares - filled_qty, "date": today_str}
                orders.append({
                    "action": "sell", "ticker": ticker, "shares": filled_qty,
                    "price": round(avg_price), "reason": reason,
                })
                continue

            logger.info("청산(%s): %s %d주 @%.0f원", reason, ticker, shares, avg_price)
            orders.append({
                "action": "sell", "ticker": ticker, "shares": shares,
                "price": round(avg_price), "reason": reason,
            })
            del holdings[ticker]
            continue

        cash += sell_floor * shares * (1 - _COMM - _TAX - _SLIP)
        logger.info("청산(%s): %s %d주 @%d원", reason, ticker, shares, sell_floor)
        orders.append({
            "action": "sell", "ticker": ticker, "shares": shares,
            "price": sell_floor, "reason": reason,
        })
        del holdings[ticker]

    portfolio["holdings"]       = holdings
    portfolio["cash"]           = round(cash, 2)
    portfolio["eod_exit_retry"] = eod_exit_retry
    _save_portfolio(portfolio)

    _write_report(ref, orders, portfolio, signals)
    # 이메일은 정확한 종가가 확정되는 15:31 run_close_report()에서 수익률과 함께 보낸다
    # (15:10은 아직 종가 확정 전이라 종가 기준 수익률을 정확히 계산할 수 없음).

    logger.info(
        "15:10 정산 완료: 청산=%d, 재시도대기=%d, 현금=%.0f원, 보유=%d종목",
        len(orders), len(eod_exit_retry), cash, len(holdings),
    )
    return {"status": "ok", "date": ref_date, "orders": orders, "cash": cash, "n_held": len(holdings)}


def check_eod_exit_retry(paper_trading: bool = True) -> dict:
    """
    1분마다 실행(15:10~15:19만 동작) — 타이머·신호청산 매도가 미체결/부분체결로
    남은 잔여 수량을, 그 시점 현재가 - N틱으로 재시도한다.

    check_buy_retry와 동일한 패턴: run_end_of_day_settlement()가 판정(hold_days/
    sell_signal_days)과 첫 시도까지 하고, 이 함수는 남은 수량만 반복 체결 시도한다.
    hold_days/sell_signal_days는 여기서 다시 건드리지 않는다 — 이미 15:10에
    하루치 판정이 끝났고, 여기는 순수하게 "체결"만 담당한다.
    """
    if paper_trading:
        return {"status": "ok", "n_retry": 0}

    portfolio = _load_portfolio()
    retry_map: dict = portfolio.get("eod_exit_retry", {})
    if not retry_map:
        return {"status": "ok", "n_retry": 0}

    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    holdings  = portfolio.get("holdings", {})
    cash      = float(portfolio.get("cash", INITIAL_CAPITAL))
    orders: list[dict] = []

    from utils.kis_client import get_order_fill_status, cancel_order, send_order_fill

    done: list[str] = []
    for ticker, info in list(retry_map.items()):
        if info.get("date") != today_str:
            logger.warning("%s 청산 재시도 정보가 오늘 것이 아님 — 폐기", ticker)
            done.append(ticker)
            continue

        if ticker not in holdings:
            # 다른 경로(예: 그 사이 손절)로 이미 청산된 경우 — 재시도 대상에서 제거
            done.append(ticker)
            continue

        reason = info["reason"]
        remaining = info["remaining"]
        quote = get_current_price(ticker)
        if quote is None or quote["current"] <= 0:
            continue  # 다음 1분 회차에 다시 시도

        sell_floor = price_minus_ticks(quote["current"], _SELL_TICKS)

        tp_odno = holdings[ticker].get("tp_odno")
        if tp_odno:
            status = get_order_fill_status(ticker, tp_odno)
            if status:
                cancel_order(
                    status.get("ord_gno_brno"), tp_odno,
                    holdings[ticker].get("tp_order_qty", remaining),
                    int(holdings[ticker].get("take_profit_price", sell_floor)),
                )
            holdings[ticker]["tp_odno"] = None
            holdings[ticker]["tp_order_qty"] = None

        fill = send_order_fill("sell", ticker, remaining, sell_floor)
        filled_qty = fill["filled_qty"]
        if filled_qty <= 0:
            continue  # 다음 1분 회차에 다시 시도

        avg_price = fill["avg_price"] or sell_floor
        cash += avg_price * filled_qty * (1 - _COMM - _TAX - _SLIP)

        if filled_qty < remaining:
            holdings[ticker]["shares"] = remaining - filled_qty
            info["remaining"] = remaining - filled_qty
            logger.warning(
                "청산(%s) 재시도 부분체결: %s %d/%d주 @%.0f원 — 잔여 %d주는 다음 회차 재시도",
                reason, ticker, filled_qty, remaining, avg_price, remaining - filled_qty,
            )
            orders.append({
                "action": "sell", "ticker": ticker, "shares": filled_qty,
                "price": round(avg_price), "reason": reason,
            })
            continue

        logger.info("청산(%s) 재시도 체결: %s %d주 @%.0f원", reason, ticker, remaining, avg_price)
        orders.append({
            "action": "sell", "ticker": ticker, "shares": remaining,
            "price": round(avg_price), "reason": reason,
        })
        del holdings[ticker]
        done.append(ticker)

    for t in done:
        del retry_map[t]

    portfolio["holdings"]       = holdings
    portfolio["cash"]           = round(cash, 2)
    portfolio["eod_exit_retry"] = retry_map
    _save_portfolio(portfolio)

    if orders:
        _write_report(pd.Timestamp(today_str), orders, portfolio)
    return {"status": "ok", "n_retry": len(retry_map)}


# ══════════════════════════════════════════════════════════════════════════
# Phase 5 — 15:31  장마감 후 종가 확정 리포트 (전일 대비·지수 대비 수익률)
# ══════════════════════════════════════════════════════════════════════════

_KOSPI_PROXY_TICKER = "069500"  # KODEX 200 — 코스피200 지수 대용
_VALUATION_HISTORY_PATH = BASE_DIR / "data" / "daily_valuation_history.json"


def _load_valuation_history() -> dict:
    if not _VALUATION_HISTORY_PATH.exists():
        return {}
    try:
        return json.loads(_VALUATION_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_valuation_history(history: dict) -> None:
    _VALUATION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _VALUATION_HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_close_report(ref_date: str | None = None) -> dict:
    """
    매일 15:31 실행 (장마감 15:30 이후). 정확한 종가가 확정된 뒤에만
    "전일 대비 수익률"과 "코스피200 대비 수익률(초과성과)"을 계산할 수 있어
    15:19 정산과 분리했다. 계산 후 그날 최종 리포트를 이메일로 발송한다.
    """
    if ref_date is None:
        ref_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    ref = pd.Timestamp(ref_date)
    date_key = ref.strftime("%Y-%m-%d")

    portfolio = _load_portfolio()
    cash      = float(portfolio.get("cash", INITIAL_CAPITAL))
    holdings  = portfolio.get("holdings", {})

    mkt_value, _ = _current_market_value(holdings)
    total_value  = cash + mkt_value

    kospi_quote = get_current_price(_KOSPI_PROXY_TICKER)
    kospi_close = kospi_quote["current"] if kospi_quote and kospi_quote.get("current", 0) > 0 else None

    history = _load_valuation_history()
    dates_before = sorted(d for d in history if d < date_key)
    prev = history.get(dates_before[-1]) if dates_before else None

    daily_return = None
    index_return = None
    excess_return = None
    if prev and prev.get("total_value"):
        daily_return = (total_value - prev["total_value"]) / prev["total_value"]
    if prev and prev.get("kospi_close") and kospi_close:
        index_return = (kospi_close - prev["kospi_close"]) / prev["kospi_close"]
    if daily_return is not None and index_return is not None:
        excess_return = daily_return - index_return

    history[date_key] = {"total_value": total_value, "kospi_close": kospi_close}
    _save_valuation_history(history)

    return_lines = ["", "## 수익률 (종가 기준)"]
    if daily_return is not None:
        return_lines.append(f"- 전일 대비: {daily_return*100:+.2f}%")
    else:
        return_lines.append("- 전일 대비: 데이터 없음 (첫날 또는 전일 기록 누락)")
    if index_return is not None:
        return_lines.append(f"- 코스피200(KODEX200) 대비: {index_return*100:+.2f}%")
    if excess_return is not None:
        return_lines.append(f"- 초과성과(알파): {excess_return*100:+.2f}%p")
    return_lines.append("")

    fname = REPORTS_DIR / f"report_{ref.strftime('%Y%m%d')}.md"
    if fname.exists():
        existing = fname.read_text(encoding="utf-8").rstrip("\n").split("\n")
    else:
        existing = [f"# 일일 매매 리포트 — {date_key}"]

    # 같은 날 재실행(서비스 재시작 등)될 경우 "## 수익률" 섹션이 중복 삽입되지
    # 않도록, 기존에 있던 섹션(다음 "## " 제목 전까지)을 먼저 제거한다.
    start = next((i for i, l in enumerate(existing) if l.startswith("## 수익률")), None)
    if start is not None:
        end = next((i for i in range(start + 1, len(existing)) if existing[i].startswith("## ")), len(existing))
        del existing[max(start - 1, 0):end]  # 섹션 앞의 빈 줄도 같이 제거

    # "## 총 평가금액" 줄 바로 뒤에 수익률 섹션을 끼워 넣는다 (리포트 맨 아래가 아니라
    # 상단, 총 평가금액 밑에 바로 보이도록).
    insert_at = len(existing)
    for i, line in enumerate(existing):
        if line.startswith("## 총 평가금액"):
            insert_at = i + 1
            break
    lines = existing[:insert_at] + return_lines + existing[insert_at:]

    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("종가 리포트 갱신: %s (일간=%s, 지수=%s)", fname, daily_return, index_return)

    _email_current_report(ref, "장 마감 최종 리포트")

    return {
        "status": "ok", "date": date_key, "total_value": total_value,
        "daily_return": daily_return, "index_return": index_return, "excess_return": excess_return,
    }
