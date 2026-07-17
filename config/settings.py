"""
단일 설정 파일 — 모든 상수·팩터·섹터 정의의 유일한 출처.
나머지 모듈은 여기서 import만 한다.

미국 주식(S&P500) 버전 — 2026-07-16 백테스팅 결정사항 반영.
국내 코스피200 버전에서 시장이 다른 부분(브로커 API, 재무데이터 소스, 유니버스,
틱사이즈, 거래시간, 거래비용)만 교체했다. 팩터 정의·라벨·WFV 구조·리스크 공식은
market-agnostic이라 그대로 이식(US_Stock_Dev_Blueprint.md 1장 참고).

주의: DART_* 관련 상수는 전부 제거됨. data/dart_collector.py, data/fetch_shares.py,
data/sector_classifier.py, data/build_universe.py, data/build_price_cache.py는
이 파일과 함께 SEC EDGAR / yfinance / Vanguard 500 Index Fund 기반으로 재작성
예정(진행 중) — 그 전까지는 위 5개 파일이 import 에러를 낸다. agents/agent_a.py는
국내 프로젝트에서 이미 폐기된 죽은 코드라 그대로 방치.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
BACKTEST_DIR = BASE_DIR / "backtest"
REPORTS_DIR  = BASE_DIR / "reports"

# ── API 키 ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")

# SEC EDGAR — API 키 불필요, 대신 실명+연락처 형식의 User-Agent 헤더 필수
# (형식 안 지키면 IP 차단 위험. https://www.sec.gov/os/webmaster-faq#developers)
SEC_EDGAR_USER_AGENT = os.getenv("SEC_EDGAR_USER_AGENT", "")

# Tiingo — 가격데이터 보조 소스. yfinance는 상장폐지·개명 종목 커버리지가 낮아서
# (2026-07-16 실측: 17개 표본 중 12%) yfinance 실패 시에만 보충용으로 호출한다.
# 무료 티어: 시간당 50회·하루 1000회·월 500종목 — https://www.tiingo.com 무료 가입.
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY", "")

# 한국투자증권(KIS) Open API — 해외주식도 국내주식과 앱키·시크릿 공용, tr_id만 다름
# (2026-07-16 공식 GitHub로 확인). 모의/실전은 앱키·계좌 완전히 분리해서 읽고
# KIS_MOCK 값으로 어느 쪽을 쓸지만 결정한다 — 국내 프로젝트와 동일 원칙.
KIS_MOCK = os.getenv("KIS_MOCK", "true").lower() == "true"  # true=모의투자, false=실전투자

KIS_MOCK_APP_KEY            = os.getenv("KIS_MOCK_APP_KEY")
KIS_MOCK_APP_SECRET         = os.getenv("KIS_MOCK_APP_SECRET")
KIS_MOCK_ACCOUNT_NO         = os.getenv("KIS_MOCK_ACCOUNT_NO")               # 모의투자 계좌번호 앞 8자리
KIS_MOCK_ACCOUNT_PRODUCT_CD = os.getenv("KIS_MOCK_ACCOUNT_PRODUCT_CD", "01")  # 모의투자 계좌번호 뒤 2자리

KIS_REAL_APP_KEY            = os.getenv("KIS_REAL_APP_KEY")
KIS_REAL_APP_SECRET         = os.getenv("KIS_REAL_APP_SECRET")
KIS_REAL_ACCOUNT_NO         = os.getenv("KIS_REAL_ACCOUNT_NO")               # 실전투자 계좌번호 앞 8자리
KIS_REAL_ACCOUNT_PRODUCT_CD = os.getenv("KIS_REAL_ACCOUNT_PRODUCT_CD", "01")  # 실전투자 계좌번호 뒤 2자리

# 키움 OpenAPI — 국내 전용 레거시. 미국 버전에선 쓸 일 없지만 agents/agent_c.py가
# 아직 이 값을 참조하고 있어(라이브 트레이딩 코드, 이번 백테스팅 단계 범위 밖)
# import 에러 방지 차원에서 자리만 유지. agent_c.py 포팅할 때 같이 정리 예정.
KIWOOM_ACCOUNT  = os.getenv("KIWOOM_ACCOUNT")

# ── 스키마 버전 (수집 데이터 컬럼 추가·변경 시 +1) ──────────────────────────
SEC_SCHEMA_VERSION = 1

# ── 데이터 수집 시작 시점 (2026-07-16 확정) ─────────────────────────────────
# 에이전트 C(가격/기술적)는 2000년부터, 에이전트 B(펀더멘털)는 2009년부터.
# SEC EDGAR의 구조화(XBRL) 재무 데이터는 대형가속제출자 XBRL 의무화 이전인
# 2009년 이전 구간은 실무적으로 신뢰할 수 없음 — 가격 데이터(yfinance)만
# 2000년부터 무료로 확보 가능.
PRICE_START_DATE   = "2000-01-01"  # yfinance 가격데이터 시작일 (Agent C)
AGENT_B_START_YEAR = 2009          # SEC EDGAR 재무데이터 시작 연도 (Agent B)

# ── 초기 자본금 (백테스팅, 2026-07-16 확정: $30,000) ────────────────────────
INITIAL_CAPITAL = 30_000  # USD

# ── 반기 교체 기준월 (1월·7월 첫째 월요일) — market-agnostic, 그대로 이식 ───
SEMIANNUAL_MONTHS = [1, 7]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 섹터 정의 — GICS 11개 섹터 (S&P가 공식적으로 쓰는 분류 체계)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 국내 KRX_TO_CUSTOM_SECTOR 같은 별도 재매핑이 필요 없음 — GICS 자체가 이미
# 깔끔한 11개 섹터라 그대로 사용(2026-07-16 결정: 과거 시점에도 현재 GICS
# 분류를 근사 적용). 위키피디아 S&P500 표의 "GICS Sector" 컬럼에서 가져온다
# (data/sector_classifier.py에서 사용 예정).
GICS_SECTORS: list[str] = [
    "Information Technology", "Health Care", "Financials",
    "Consumer Discretionary", "Communication Services", "Industrials",
    "Consumer Staples", "Energy", "Utilities", "Real Estate", "Materials",
]

# 섹터별 최대 선정 수 규칙 (≤4: 1개, 5~10: 2개, >10: 무제한) — market-agnostic
def get_sector_cap(n_tickers: int) -> int | None:
    if n_tickers <= 4:
        return 1
    if n_tickers <= 10:
        return 2
    return None  # 무제한

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 에이전트 A 팩터 정의 — 국내 프로젝트에서 이미 폐기(main.py가 agent_a를 아예
# import하지 않음, 2026-07-16 확인). 실제로 쓰이지 않지만 backtest/wfv.py,
# ic_screening.py, xgb_ranker.py 등 공용 파이프라인 코드가 AGENT_A_FEATURES를
# import하므로 자리만 유지 — 값 자체는 market-agnostic한 팩터명이라 손대지 않음.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AGENT_A_FACTOR_META: list[tuple[str, bool, str]] = [
    # 수익성
    ("roe",              True,  "sector"),
    ("roa",              True,  "sector"),
    ("roic",             True,  "sector"),
    ("net_margin",       True,  "sector"),
    ("op_margin",        True,  "sector"),
    ("roe_5y_avg",       True,  "sector"),
    ("eps_per_share",    True,  "sector"),
    # 밸류에이션
    ("per",              False, "sector"),  # 낮을수록 좋음
    ("pbr",              False, "sector"),  # 낮을수록 좋음
    ("dividend_yield",   True,  "all"),
    # 효율성
    ("asset_turnover",   True,  "sector"),
    # 재무건전성
    ("debt_ratio",       False, "sector"),  # 낮을수록 좋음
    ("interest_coverage",True,  "sector"),
    ("current_ratio",    True,  "sector"),
    # 현금흐름
    ("fcf_margin",       True,  "all"),
    ("fcf_yield",        True,  "all"),
    ("cash_ratio",       True,  "all"),
    # 성장성
    ("revenue_growth",   True,  "all"),
    ("eps_growth_5y",    True,  "all"),
    ("equity_growth_5y", True,  "all"),
]

AGENT_A_FACTORS_CONTINUOUS = [m[0] for m in AGENT_A_FACTOR_META]

AGENT_A_FACTORS_BINARY = [
    "fcf_positive", "profit_streak_5y", "equity_positive", "no_value_trap",
]

AGENT_A_FACTORS_TIMESERIES = [
    "sector_momentum", "days_since_earnings", "ear_3d_final", "ear_3d_trend",
]

# 최종 피처 목록 (XGBoost 입력 순서 고정)
AGENT_A_FEATURES: list[str] = (
    [f"{f}_percentile" for f in AGENT_A_FACTORS_CONTINUOUS]
    + [f"{f}_cross_z"   for f in AGENT_A_FACTORS_CONTINUOUS]
    + [f"{f}_time_z"    for f in AGENT_A_FACTORS_CONTINUOUS]
    + AGENT_A_FACTORS_BINARY
    + AGENT_A_FACTORS_TIMESERIES
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 에이전트 B 팩터 정의 — 실제 매매에 쓰이는 유일한 펀더멘털 에이전트.
# 팩터 공식 자체는 market-agnostic이라 국내와 동일하게 이식.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AGENT_B_FACTOR_META: list[tuple[str, bool, str]] = [
    # 수익성
    ("roic",               True,  "sector"),
    # 성장률
    ("revenue_growth",     True,  "all"),
    ("op_profit_growth",   True,  "all"),
    ("eps_growth",         True,  "all"),
    ("revenue_cagr_5y",    True,  "all"),
    ("eps_cagr_5y",        True,  "all"),
    ("gross_margin_trend", True,  "all"),
    ("operating_leverage", True,  "all"),
    # R&D·투자
    ("rd_ratio",           True,  "all"),
    ("employee_growth",    True,  "all"),
    # 밸류에이션
    ("peg",                False, "sector"),  # 낮을수록 좋음
    ("psr",                False, "sector"),  # 낮을수록 좋음
    ("cash_to_mktcap",     True,  "all"),
    # 모멘텀
    ("momentum_3m",        True,  "all"),
    ("momentum_6m",        True,  "all"),
    ("high52w_pct",        True,  "all"),
]

AGENT_B_FACTORS_CONTINUOUS = [m[0] for m in AGENT_B_FACTOR_META]

AGENT_B_FACTORS_BINARY = [
    "profit_streak_2y", "equity_positive",
]

AGENT_B_FACTORS_TIMESERIES = [
    "sector_momentum", "days_since_earnings", "ear_3d_final", "ear_3d_trend",
]

AGENT_B_FEATURES: list[str] = (
    [f"{f}_percentile" for f in AGENT_B_FACTORS_CONTINUOUS]
    + [f"{f}_cross_z"   for f in AGENT_B_FACTORS_CONTINUOUS]
    + [f"{f}_time_z"    for f in AGENT_B_FACTORS_CONTINUOUS]
    + AGENT_B_FACTORS_BINARY
    + AGENT_B_FACTORS_TIMESERIES
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 에이전트 C (CNN-1D + LSTM) 설정 — market-agnostic, 그대로 이식
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CNN_CONFIG = {
    "window":   100,  # 2026-07-17 최종 확정: 3클래스 10일 → 이진분류 100영업일로 변경
    "channels": ["open", "high", "low", "close", "volume"],
}

LSTM_FEATURES: list[str] = [
    "ma_alignment", "macd_score", "close_vs_ema20", "close_vs_ema60",
    "rsi_norm", "stoch_norm", "bb_pos",
    "adx_norm", "vol_ratio", "obv_norm",
    "close_vs_ma200", "high52w_pos", "week_return", "price_vs_bb_width",
    "volume_price_trend",
    "gap_ratio", "sector_relative_strength",
]

LSTM_CONFIG = {
    "window":     100,  # 2026-07-17 최종 확정: 20일 → 이진분류 100영업일로 변경 (CNN과 동일)
    "n_features": len(LSTM_FEATURES),  # 17
    "n_classes":  3,                   # Buy=0, Hold=1, Sell=2 — 3클래스 경로 기본값, 이진 경로는 run_c_wfv_bin이 n_classes=2로 별도 오버라이드
    "min_epochs": 200,
}

# 라벨 파라미터 (CNN·LSTM 공통) — market-agnostic, 그대로 이식
LABEL_CONFIG = {
    "N":                 10,   # 라벨 기간 (영업일)
    "K":                 1.0,  # ATR 배수
    "smooth_days":       5,    # 평활화 윈도우 크기 (T+61~T+65)
    "label_cutoff_days": 65,   # WFV fold 경계 누수 방지 버퍼
}

# α+β=1.0 제약, α ∈ {0.0, 0.1, ..., 1.0}
ALPHA_GRID = [round(i * 0.1, 1) for i in range(11)]

# ── 아키텍처 탐색 범위 (WFV 전 소규모 실험으로 후보 선정) — market-agnostic ──
CNN_ARCH_SEARCH = {
    "num_conv_layers": [2, 3],
    "num_filters":     [32, 64],
    "kernel_size":     [3, 5],
    "dropout":         [0.2, 0.3],
}

LSTM_ARCH_SEARCH = {
    "hidden_size":   [64, 128],
    "num_layers":    [1, 2],
    "dropout":       [0.2, 0.3],
    "bidirectional": False,   # 미래 데이터 누수 구조적 차단
}

XGB_PARAM_SEARCH = {
    "n_estimators":          [200, 500, 1000],
    "learning_rate":         [0.01, 0.05, 0.1],
    "max_depth":             [4, 6, 8],
    "subsample":             [0.7, 0.9],
    "colsample_bytree":      [0.7, 0.9],
    "min_child_weight":      [5, 10],
    "objective":             "rank:ndcg",
    "tree_method":           "hist",
    "early_stopping_rounds": 50,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IC 스크리닝 — market-agnostic, 그대로 이식
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IC_MIN_THRESHOLD = 0.002  # |IC| 최솟값
IC_MIN_FOLDS     = 2      # 통과해야 할 최소 fold 수 (5-fold 중)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WFV 설정 (확장 윈도우)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WFV_CONFIG = {
    "window_type":    "expanding",
    "ab_train_start": AGENT_B_START_YEAR,  # 2009 — 키 이름은 국내 코드 호환용으로 유지, 실제로는 B 학습 시작연도
    "c_train_start":  2000,                # Agent C 학습 시작연도
    "test_window":    1,
    "n_folds":        13,
    # TODO: 실제 SEC EDGAR/yfinance 데이터 커버리지 확인 후 test_years·pass_threshold
    # 재조정 필요 — 지금은 "b_train_start(2009) + 4년 워밍업"으로 임시 설정한 값.
    "test_years":     [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    "pass_threshold": 10,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 매매·리스크 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRADE_CONFIG = {
    "max_hold_days":      10,   # 타이머 청산 (영업일) — market-agnostic
    # 진입: final_score >= buy_upper → 익일 시가
    "gap_atr_window":     10,   # 갭 필터 ATR 기간
    "gap_filter_mult":    1.5,  # gap > ATR × 1.5 → 당일 스킵
    # 손절: 과거 N일 Low 최솟값 × (1 - buffer)
    "swing_low_window":   10,
    "swing_low_buffer":   0.01,
    # 익절: entry + (entry - SL) × R:R
    "take_profit_rr":     2.0,

    # ── 장중 실시간 체결 관련 (NYSE/NASDAQ 정규장 09:30~16:00 US/Eastern 기준) ──
    # 국내는 KST 단일 시간대라 서머타임이 없었지만, 미국은 3월·11월 연 2회
    # 서머타임 전환이 있다 — 이 표는 "거래소 현지시각" 기준값이고, 서버가
    # 실제로 이 시각에 맞춰 동작하게 하는 서머타임 처리(main.py 스케줄러)는
    # 실운용 전환 단계에서 별도로 다룰 예정(지금은 백테스팅 범위 밖).
    "buy_tick_offset":       2,      # 매수: 현재가 + 2틱 (체결 보장, 예산도 이 가격 기준)
    "sell_tick_offset":      3,      # 매도(손절·타이머): 현재가 - 3틱 (체결 보장)
    "intraday_check_interval_min": 1,  # 장중 손절 감시 주기 (분)
    "market_open_time":      "09:30",
    "entry_time":            "09:31",  # 실시간 시가 확인 + 갭 필터 + 매수 실행
    "timer_exit_time":       "15:50",  # 10영업일 타이머 청산 (마감 10분 전)
    "market_close_time":     "16:00",
}

# 미국 주식 틱사이즈(2026-07-16 확정): $1 이상 $0.01 균일, $1 미만 $0.0001.
# (참고: SEC가 유동성 높은 종목 대상 $0.005 틱 신설 개정안을 2026-11 첫 영업일
# 시행 예정으로 유예해둠 — 실거래 전환 시점에 재확인 필요.)
TICK_SIZE_TABLE: list[tuple[float, float]] = [
    (1.0,          0.0001),
    (float("inf"), 0.01),
]

# WFV 완료 후 신호 분포 기반으로 재설정 예정 — market-agnostic
SIGNAL_THRESHOLDS = {
    "buy_upper":        0.25,  # 이 이상이면 익일 시가 진입 (WFV 후 분포 기반 재설정)
    "sell_consecutive": 4,
}

# 거래비용 — 한국투자증권 해외주식 공식 수수료 페이지 확인 완료(2026-07-16).
# 국내와 달리 별도 거래세(tax) 없음 — 대신 매도 시에만 SEC Fee + FINRA TAF 부과.
TRANSACTION_COSTS = {
    "commission":           0.0025,    # 0.25% (매수·매도 동일, 최소수수료 없음)
    "sec_fee":               0.0000206, # 매도 시 0.00206% (Section 31 수수료)
    "finra_taf_per_share":   0.000195,  # 매도 시 주당 $0.000195 (2026-01-01 개정 요율)
    "finra_taf_cap":         9.79,      # 건당 최대 (이 계좌 규모에선 사실상 발동 안 함)
    "slippage":              0.001,     # 0.1% (국내와 동일 가정, 실데이터로 추후 검증)
}

RISK_LIMITS = {
    "target_position_pct":  0.11,   # 목표 비중 11% (v2) — market-agnostic
    "budget_range_pct":     0.03,   # 허용 범위 +3% → budget_high = low + 3%
    "max_single_position":  0.15,   # 1회 매수 상한 15% (1주 예외 허용)
    "skip_single_pct":      0.40,   # 1주 가격 > 40% → 건너뜀
    "max_sector_position":  0.50,   # 섹터 최대 비중
    "max_daily_turnover":   0.30,
}

# 거래정지 장기화 알림 기준 (영업일) — market-agnostic
TRADING_HALT_ALERT_DAYS = 5

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEC EDGAR API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEC_EDGAR_BASE_URL   = "https://data.sec.gov"
SEC_EDGAR_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar"

# 10-Q/10-K만 추적(2026-07-16 확정) — 8-K 잠정실적 발표는 별도 추적 안 함.
# form 값은 SEC company facts API의 "form" 필드와 매칭.
SEC_FORM_TYPES = ["10-K", "10-Q"]

# ── 유니버스: Vanguard 500 Index Fund SEC 반기보고서 (2026-07-16 확정) ──────
# CIK 0000036405 (VANGUARD INDEX FUNDS, 옛 이름 VANGUARD INDEX TRUST/
# FIRST INDEX INVESTMENT TRUST) — 세계 최초 인덱스펀드. N-30D(1997~2003)/
# N-CSR(2003~) 반기보고서의 "Statement of Net Assets"에 500 Index Fund
# 보유종목 전체가 회사명 기준으로 나열되어 있음 — 2000년부터 지금까지 연 2회
# (3월·8월 제출) 스냅샷으로 커버 가능. 위키피디아 방식(2005-09-14 이전 재구성
# 불가) 대신 이 방식으로 확정. 회사명→티커 매칭이 별도로 필요(구현 예정).
VANGUARD_500_CIK = "0000036405"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 매크로 지표 캐시 TTL (일 기준) — 국내(usd_krw/kr_cli/kr_indprod) 항목 제거,
# 환전은 사장이 직접 수동 처리하기로 결정(2026-07-16)해 환율 리스크 모델링 불필요.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MACRO_CACHE_TTL = {
    "vix":        1,
    "hy_spread":  1,
    "us_10y":     7,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 키움 OpenAPI — 국내 전용 레거시, agents/agent_c.py 정리 전까지 자리만 유지
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KIWOOM_CONFIG = {
    "account":     KIWOOM_ACCOUNT,
    "order_type":  "지정가",
    "market_code": "KSP",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 한국투자증권(KIS) Open API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# KIS_MOCK 값에 따라 모의투자/실전투자 세트 중 실제 사용할 것만 골라 담는다.
# 나머지 코드(utils/kis_client.py 등)는 이 KIS_CONFIG만 보고 동작하므로
# 어느 쪽 자격증명이 실제로 쓰이는지 헷갈릴 여지가 없다.
# 해외주식 주문은 거래소코드(NASD/NYSE 등)가 추가로 필요 — kis_client.py
# 재작성 시(실운용 단계) 반영 예정, 지금은 백테스팅 범위 밖.
KIS_CONFIG = {
    "app_key":            KIS_MOCK_APP_KEY if KIS_MOCK else KIS_REAL_APP_KEY,
    "app_secret":         KIS_MOCK_APP_SECRET if KIS_MOCK else KIS_REAL_APP_SECRET,
    "account_no":         KIS_MOCK_ACCOUNT_NO if KIS_MOCK else KIS_REAL_ACCOUNT_NO,
    "account_product_cd": KIS_MOCK_ACCOUNT_PRODUCT_CD if KIS_MOCK else KIS_REAL_ACCOUNT_PRODUCT_CD,
    "mock":               KIS_MOCK,   # True=모의투자, False=실전투자
    "order_type":         "지정가",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 시계열 Z-score 설정 — market-agnostic, 그대로 이식
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIME_Z_QUARTERS_PREFERRED = 8  # 권장: 8분기(2년)
TIME_Z_QUARTERS_MIN       = 4  # 최소: 4분기(1년)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파이프라인 시작 시 일관성 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_feature_lists() -> None:
    """settings.py 내부 정의 간 일관성 검증."""
    assert len(AGENT_A_FEATURES) == (
        len(AGENT_A_FACTORS_CONTINUOUS) * 3
        + len(AGENT_A_FACTORS_BINARY)
        + len(AGENT_A_FACTORS_TIMESERIES)
    ), "AGENT_A_FEATURES 길이 불일치"

    assert len(AGENT_B_FEATURES) == (
        len(AGENT_B_FACTORS_CONTINUOUS) * 3
        + len(AGENT_B_FACTORS_BINARY)
        + len(AGENT_B_FACTORS_TIMESERIES)
    ), "AGENT_B_FEATURES 길이 불일치"

    assert len(LSTM_FEATURES) == LSTM_CONFIG["n_features"], (
        f"LSTM_FEATURES({len(LSTM_FEATURES)}) ≠ n_features({LSTM_CONFIG['n_features']})"
    )

    assert all(a + b == 1.0 for a in ALPHA_GRID for b in [round(1.0 - a, 1)]), (
        "α+β=1.0 제약 위반"
    )

    assert len(WFV_CONFIG["test_years"]) == WFV_CONFIG["n_folds"], (
        "WFV_CONFIG test_years/n_folds 길이 불일치"
    )


validate_feature_lists()
