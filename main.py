"""
퀀트멘탈 시스템 진입점.

실행:
  python main.py --mode live       # 실거래 (매일 자동 실행)
  python main.py --mode paper      # 페이퍼트레이딩
  python main.py --mode backtest   # WFV 백테스팅
  python main.py --mode daily      # 오늘 날짜로 일회성 실행 (테스트용)

스케줄:
  매일 8:00     — Agent A/B 부분 업데이트 + Agent C 매매 신호·주문
  금요일 15:30  — Agent A/B 전체 재랭킹 + MetaML 스냅샷 저장
  반기 첫 월요일 — KOSPI200 리밸런싱 + watchlist 교체
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path

import pandas as pd
import schedule

# 설정 로드 (settings 임포트 = 유효성 검증 실행)
# 에이전트 A는 더 이상 사용하지 않는다 — 실제 매매(Agent C)는 항상 Agent B의
# factor_dataset_B.parquet + 학습된 모델만 사용했고, A의 워치리스트는 어디서도
# 소비되지 않는 산출물이었다. 유니버스도 가치/성장 섹터 분리 없이 코스피200
# 전체를 Agent B 하나가 담당하도록 통합했다.
from config.settings import BACKTEST_DIR, BASE_DIR, SEMIANNUAL_MONTHS
from agents.agent_b import (
    friday_rerank_agent_b,
    run_agent_b,
    update_scores_agent_b,
)
from agents.agent_c import (
    check_buy_retry,
    check_eod_exit_retry,
    check_intraday_stop_loss,
    check_model_status,
    check_tp_fills,
    run_close_report,
    run_daily_signal,
    run_end_of_day_settlement,
    run_market_open_entry,
)
from backtest.snapshot_manager import fill_snapshot_labels, save_weekly_snapshot
from data.build_factor_dataset import fill_labels
from data.build_price_cache import build_price_cache
from data.build_universe import get_universe_by_date
from data.dart_collector import run_collection as run_dart_collection
from utils.calendar_utils import (
    get_semiannual_dates,
    is_trading_day,
)
from utils.logger import get_logger

logger = get_logger(__name__)

PORTFOLIO_PATH = BASE_DIR / "portfolio_state.json"


# ── 장 시작 전 데이터 최신화 ───────────────────────────────────────────────────

def _get_today_tickers(ref_date_str: str) -> list[str]:
    """
    오늘 점수 계산·매매에 필요한 종목 목록.

    네 군데를 합집합으로 모은다 — 하나라도 빠지면 그 종목은 가격이
    안 갱신되어 "오늘 시가"를 못 읽고 매수/청산 판단이 조용히 스킵된다
    (실제로 factor_dataset_B 풀 종목이 유니버스에는 없어서 겪은 문제).
      1. 현재 KOSPI200(시총 기준) 유니버스 — Agent A/B 점수 계산 대상
      2. 보유 종목 — 청산 판단에 매일 필요
      3. 매수 대기 종목(pending_buys) — 내일 시가 체결 판단에 필요
      4. Agent C가 실제로 뽑는 Agent B 상위 풀의 원본 유니버스
         (factor_dataset_B.parquet 전체 티커) — 위 KOSPI200 유니버스와
         구성이 다를 수 있어 별도로 포함해야 한다.
    """
    tickers: set[str] = set()
    try:
        tickers.update(get_universe_by_date(ref_date_str))
    except RuntimeError as e:
        logger.error("유니버스 로드 실패 — 가격/재무 갱신 범위 축소: %s", e)

    if PORTFOLIO_PATH.exists():
        try:
            state = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
            tickers.update(state.get("holdings", {}).keys())
            tickers.update(state.get("pending_buys", {}).keys())
        except Exception as e:
            logger.warning("portfolio_state.json 로드 실패: %s", e)

    factor_b_path = BACKTEST_DIR / "results" / "factor_dataset_B.parquet"
    if factor_b_path.exists():
        try:
            import pandas as pd
            tickers.update(pd.read_parquet(factor_b_path, columns=["ticker"])["ticker"].unique().tolist())
        except Exception as e:
            logger.warning("factor_dataset_B.parquet 티커 로드 실패: %s", e)

    return sorted(tickers)


def refresh_market_data(ref_date_str: str) -> None:
    """
    매일 8:00 루틴 최초 단계 — 가격·재무 데이터를 최신화한다.
    이 단계가 없으면 Agent A/B/C가 예전 캐시(오래된 가격·재무)로
    "오늘 계산"을 수행하게 되어 실제로는 며칠~몇 주 전 기준 판단이 된다.
    """
    tickers = _get_today_tickers(ref_date_str)
    if not tickers:
        logger.warning("갱신 대상 종목 없음 — 가격/재무 데이터 갱신 스킵")
        return

    logger.info("가격 데이터 갱신 시작: %d종목", len(tickers))
    try:
        build_price_cache(tickers=tickers)
    except Exception as e:
        logger.error("가격 데이터 갱신 실패: %s", e)

    logger.info("DART 재무 데이터 갱신 시작: %d종목", len(tickers))
    try:
        run_dart_collection(tickers=tickers)
    except Exception as e:
        logger.error("DART 재무 데이터 갱신 실패: %s", e)


# ── 스케줄 작업 함수 ──────────────────────────────────────────────────────────

def _is_friday(dt: pd.Timestamp) -> bool:
    return dt.dayofweek == 4


def _is_semiannual_monday(dt: pd.Timestamp) -> bool:
    """1월·7월 첫째 월요일 여부."""
    if dt.dayofweek != 0:  # 월요일이 아님
        return False
    if dt.month not in SEMIANNUAL_MONTHS:
        return False
    dates = get_semiannual_dates(dt.year)
    return any(dt.date() == d.date() for d in dates)


def job_daily_800(alpha: float = 0.5) -> None:
    """
    매일 8:00 루틴 — 전일 종가 기준 계산만 수행 (실제 체결은 09:01에 별도 진행).
    1. 가격·재무 데이터 최신화
    2. 반기 리밸런싱 여부 판단 / Agent B 부분 업데이트 (3일 대기 관리 포함)
    3. Agent C 매수 후보 계산 (run_daily_signal)
    """
    today = pd.Timestamp.today().normalize()

    if not is_trading_day(today):
        logger.info("휴장일 — 스킵: %s", today.strftime("%Y-%m-%d"))
        return

    logger.info("=== 08:00 루틴 시작: %s ===", today.strftime("%Y-%m-%d"))

    # 0. 가격·재무 데이터 최신화 (반드시 점수 계산보다 먼저)
    refresh_market_data(today.strftime("%Y-%m-%d"))

    # 반기 첫 월요일: 리밸런싱
    # Agent B 실패가 있어도 Agent C(실제 매매)는 반드시 진행되어야 하므로 격리한다.
    try:
        if _is_semiannual_monday(today):
            logger.info("반기 리밸런싱 실행")
            run_agent_b(today.strftime("%Y-%m-%d"))
        else:
            # 일반 영업일: 3일 대기 관리 + 부분 업데이트
            update_scores_agent_b(today.strftime("%Y-%m-%d"))
    except Exception as e:
        logger.error("Agent B 업데이트 실패 — Agent C는 계속 진행: %s", e)

    # Agent C: 매수 후보 계산 (실행은 09:01 job_market_open_entry에서)
    result = run_daily_signal(ref_date=today.strftime("%Y-%m-%d"), alpha=alpha)
    logger.info("08:00 신호 계산 완료: %s", result.get("status"))


def job_market_open_entry(paper_trading: bool = True) -> None:
    """09:01 루틴 — 실시간 시가 확인 + 매수 실행 + 보유종목 익절주문 재등록."""
    today = pd.Timestamp.today().normalize()
    if not is_trading_day(today):
        return
    result = run_market_open_entry(ref_date=today.strftime("%Y-%m-%d"), paper_trading=paper_trading)
    logger.info("09:01 진입 완료: %s", result.get("status"))


def job_morning_report() -> None:
    """
    09:10 루틴 — 그날치 리포트 이메일 발송.
    09:01 직후 바로 보내면 매수 재시도(check_buy_retry, 1분마다)로 몇 분 뒤에나
    체결되는 종목이 리포트에서 빠진다(2026-07-13 실제 확인 — 9종목 중 7종목이
    재시도로 체결). 재시도가 어느 정도 끝난 09:10까지 기다렸다가 그때까지
    누적된 리포트 파일을 보낸다.
    """
    today = pd.Timestamp.today().normalize()
    if not is_trading_day(today):
        return
    from agents.agent_c import _email_current_report
    _email_current_report(today, "장 시작 매수·익절주문 완료")


def job_intraday_stop_loss(paper_trading: bool = True) -> None:
    """
    1분마다 호출 — 실제로는 거래일 09:00~15:19 사이에만 동작.
    15:20~15:30은 KRX 종가 단일가매매(동시호가) 구간이라 실시간 연속 체결이
    아니라 15:30에 한 번에 모아 처리되므로, "현재가 보고 즉시 대응"하는
    손절 로직은 연속거래가 확실히 살아있는 15:19까지만 돌린다.
    """
    today = pd.Timestamp.today().normalize()
    if not is_trading_day(today):
        return
    now_t = datetime.now().time()
    if not (dtime(9, 0) <= now_t <= dtime(15, 19)):
        return
    check_intraday_stop_loss(paper_trading=paper_trading)


def job_buy_retry(paper_trading: bool = True) -> None:
    """1분마다 호출 — 09:01 매수 중 미체결/부분체결로 남은 잔여 수량을 재시도. 09:00~15:19만 동작."""
    today = pd.Timestamp.today().normalize()
    if not is_trading_day(today):
        return
    now_t = datetime.now().time()
    if not (dtime(9, 0) <= now_t <= dtime(15, 19)):
        return
    check_buy_retry(paper_trading=paper_trading)


def job_check_tp_fills(paper_trading: bool = True) -> None:
    """1분마다 호출 — 걸어둔 익절 지정가 주문의 부분/전량 체결 여부 확인. 09:00~15:19만 동작."""
    today = pd.Timestamp.today().normalize()
    if not is_trading_day(today):
        return
    now_t = datetime.now().time()
    if not (dtime(9, 0) <= now_t <= dtime(15, 19)):
        return
    check_tp_fills(paper_trading=paper_trading)


def job_eod_settlement(paper_trading: bool = True, alpha: float = 0.5) -> None:
    """
    15:10 루틴 — 타이머(10영업일)·4일 연속 신호청산 판정 + 첫 매도 시도.
    미체결/부분체결분은 check_eod_exit_retry가 15:19까지 1분마다 재시도한다.
    15:20부터는 KRX 종가 단일가매매(동시호가) 구간이라 연속거래가 아니므로,
    실시간 체결을 기대하는 청산 주문은 그 전(15:19)까지 끝내야 한다.
    """
    today = pd.Timestamp.today().normalize()
    if not is_trading_day(today):
        return
    result = run_end_of_day_settlement(
        ref_date=today.strftime("%Y-%m-%d"), alpha=alpha, paper_trading=paper_trading
    )
    logger.info("15:10 정산 완료: %s", result.get("status"))


def job_eod_exit_retry(paper_trading: bool = True) -> None:
    """1분마다 호출 — 15:10 정산에서 미체결로 남은 타이머/신호청산 잔량 재시도. 15:10~15:19만 동작."""
    today = pd.Timestamp.today().normalize()
    if not is_trading_day(today):
        return
    now_t = datetime.now().time()
    if not (dtime(15, 10) <= now_t <= dtime(15, 19)):
        return
    check_eod_exit_retry(paper_trading=paper_trading)


def job_close_report() -> None:
    """15:31 루틴 — 장마감(15:30) 후 종가 확정 리포트(전일 대비·지수 대비 수익률) + 이메일."""
    today = pd.Timestamp.today().normalize()
    if not is_trading_day(today):
        return
    result = run_close_report(ref_date=today.strftime("%Y-%m-%d"))
    logger.info("15:31 종가 리포트 완료: %s", result.get("status"))


def job_friday_1530(alpha: float = 0.5) -> None:
    """
    금요일 15:30 루틴.
    4-A: 전체 재랭킹 (Agent B가 코스피200 전체를 재계산 → factor_dataset_B.parquet 갱신)
    4-B: MetaML 스냅샷 저장
    4-C: label_3m 사후 채움
    """
    today = pd.Timestamp.today().normalize()

    if not is_trading_day(today) or not _is_friday(today):
        return

    logger.info("=== 금요일 15:30 루틴 시작: %s ===", today.strftime("%Y-%m-%d"))
    ref_date_str = today.strftime("%Y-%m-%d")

    # 4-A: 전체 재랭킹
    # 실패해도 스케줄러 프로세스 자체는 죽지 않도록 격리 — 다음 월요일 매매(Agent C)는
    # 이 실패와 무관하게 정상 진행되어야 한다.
    try:
        top_b = friday_rerank_agent_b(ref_date_str)
    except Exception as e:
        logger.error("Agent B 금요일 전체 재랭킹 실패 — 스킵: %s", e)
        top_b = []

    # 4-B: MetaML 스냅샷 저장
    try:
        from agents.agent_c import compute_signals
        from models.cnn_model import load_cnn
        from models.lstm_model import load_lstm

        arch_params_path = __import__("pathlib").Path("models/saved/arch_params.json")
        arch = {}
        if arch_params_path.exists():
            import json
            arch = json.loads(arch_params_path.read_text(encoding="utf-8"))

        cnn_m  = load_cnn(arch.get("cnn"))
        lstm_m = load_lstm(arch.get("lstm"))

        sigs = compute_signals(top_b, today, alpha, cnn_m, lstm_m)

        # Agent A 폐기 — scores_a는 빈 DataFrame으로 전달
        scores_a = pd.DataFrame(columns=["ticker", "xgb_score"])
        scores_b = pd.DataFrame({"ticker": top_b, "xgb_score": [0.5] * len(top_b)})

        import json
        portfolio = json.loads(
            (__import__("pathlib").Path("portfolio_state.json")).read_text(encoding="utf-8")
        ) if __import__("pathlib").Path("portfolio_state.json").exists() else {}
        held = list(portfolio.get("holdings", {}).keys())

        save_weekly_snapshot(today, scores_a, scores_b, sigs, held)
    except Exception as e:
        logger.error("MetaML 스냅샷 저장 실패: %s", e)

    # 4-C: label_3m 사후 채움 (MetaML용 weekly_snapshots)
    try:
        fill_snapshot_labels()
    except Exception as e:
        logger.error("label_3m 채움 실패 (weekly_snapshots): %s", e)

    # 4-C': factor_dataset_B.parquet 자체 label_3m 사후 채움 — 실제 XGBoost B
    # 학습 데이터. weekly_snapshots(MetaML용)와는 별개 파일이라 별도로 채워야 하는데
    # 지금까지 이 호출이 빠져 있어서 2026-03-13 이후 실운용 누적분 라벨이 계속
    # 비어있었다(2026-07-09 발견). fill_labels() 내부에 이미 65영업일 경과 여부를
    # 체크하는 안전장치가 있어 아직 라벨을 낼 수 없는(미래 가격이 필요한) 스냅샷은
    # 자동으로 건너뛴다 — 룩어헤드 위험 없음.
    try:
        fill_labels("B")
    except Exception as e:
        logger.error("label_3m 채움 실패 (factor_dataset_B): %s", e)

    logger.info("=== 금요일 15:30 루틴 완료 ===")


# ── 실행 모드 ─────────────────────────────────────────────────────────────────

def job_minute_bar_collection() -> None:
    """
    15:35 루틴 — 코스피200(+지수 ETF 대용) 당일 분봉 수집.
    매매 시스템과 무관 — 다른 프로그램에서 재사용할 데이터 축적용.
    실패해도 매매 로직에 영향 없도록 완전히 격리한다.
    """
    today = pd.Timestamp.today().normalize()
    if not is_trading_day(today):
        return
    try:
        from data.collect_minute_bars import run_daily_minute_bar_collection
        result = run_daily_minute_bar_collection(today.strftime("%Y-%m-%d"))
        logger.info("분봉 수집 완료: %s", result)
    except Exception as e:
        logger.error("분봉 수집 실패: %s", e)


def run_live(paper_trading: bool = True, alpha: float = 0.5) -> None:
    """실거래/페이퍼트레이딩 — 스케줄러 무한 루프. KRX 본장(09:00~15:30) 기준."""
    logger.info("시스템 시작 (paper=%s, alpha=%.1f)", paper_trading, alpha)
    check_model_status()

    schedule.every().day.at("08:00").do(job_daily_800, alpha=alpha)
    schedule.every().day.at("09:01").do(job_market_open_entry, paper_trading=paper_trading)
    schedule.every().day.at("09:10").do(job_morning_report)
    schedule.every(1).minutes.do(job_intraday_stop_loss, paper_trading=paper_trading)
    schedule.every(1).minutes.do(job_buy_retry, paper_trading=paper_trading)
    schedule.every(1).minutes.do(job_check_tp_fills, paper_trading=paper_trading)
    schedule.every().day.at("15:10").do(job_eod_settlement, paper_trading=paper_trading, alpha=alpha)
    schedule.every(1).minutes.do(job_eod_exit_retry, paper_trading=paper_trading)
    schedule.every().day.at("15:31").do(job_close_report)
    # 분봉 수집 대상은 정규장(09:00~15:30) 데이터뿐이라 시간외단일가(~18:00)까지
    # 기다릴 필요가 없다 — 마감 5분 후면 그날 분봉이 이미 다 확정되어 있다.
    # 클라우드 서버를 장중에만 켜두는 구조라, 이 시각이 늦어질수록
    # 서버를 더 오래 켜둬야 해서(=비용) 최대한 당겨둔다.
    schedule.every().day.at("15:35").do(job_minute_bar_collection)
    schedule.every().friday.at("15:30").do(job_friday_1530, alpha=alpha)

    logger.info("스케줄러 등록 완료 (08:00/09:01/09:10/1분마다(~15:19)/15:10/15:10~15:19(1분마다)/15:31/15:35/금 15:30). 대기 중...")
    while True:
        schedule.run_pending()
        time.sleep(15)


def run_daily_once(date: str | None = None, paper_trading: bool = True, alpha: float = 0.5) -> None:
    """오늘 날짜로 전체 파이프라인 1회 실행 (테스트용) — 장중 루프는 1회만 호출."""
    if date is None:
        date = pd.Timestamp.today().strftime("%Y-%m-%d")
    today = pd.Timestamp(date)

    logger.info("일회성 실행: %s", date)
    check_model_status()
    refresh_market_data(date)

    # job_daily_800과 동일하게 Agent B 실패를 격리 — 테스트 경로도 실거래 스케줄과
    # 동일한 안전장치를 갖춰야 실제 동작을 신뢰성 있게 재현할 수 있다.
    try:
        if _is_semiannual_monday(today):
            run_agent_b(date)
        else:
            update_scores_agent_b(date)
    except Exception as e:
        logger.error("Agent B 업데이트 실패 — 이후 단계는 계속 진행: %s", e)

    run_daily_signal(ref_date=date, alpha=alpha)
    run_market_open_entry(ref_date=date, paper_trading=paper_trading)
    check_intraday_stop_loss(paper_trading=paper_trading)
    run_end_of_day_settlement(ref_date=date, alpha=alpha, paper_trading=paper_trading)
    check_eod_exit_retry(paper_trading=paper_trading)

    if _is_friday(today):
        job_friday_1530(alpha=alpha)


def run_backtest_pipeline() -> None:
    """WFV 백테스팅 전체 파이프라인."""
    from data.build_factor_dataset import build_factor_dataset, fill_labels
    from backtest.ic_screening import run_screening
    from models.xgb_ranker import run_wfv
    from models.train_c import run_c_wfv

    logger.info("=== 백테스팅 파이프라인 시작 ===")

    # 에이전트 A는 폐기됨 — Agent B만 처리
    # Phase 2-1: factor_dataset 생성
    logger.info("factor_dataset 생성 (B)")
    build_factor_dataset("B")

    # Phase 2-2: 라벨 채움
    logger.info("label_3m 채움 (B)")
    fill_labels("B")

    # Phase 2-3: IC 스크리닝
    logger.info("IC 스크리닝")
    run_screening("B")

    # Phase 2-4: XGBoost WFV
    logger.info("XGBoost B WFV")
    run_wfv("B")

    # Phase 2-5: CNN+LSTM WFV
    logger.info("Agent C (CNN+LSTM) WFV")
    run_c_wfv()

    logger.info("=== 백테스팅 파이프라인 완료 ===")


# ── CLI 진입점 ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="퀀트멘탈 실행")
    parser.add_argument(
        "--mode",
        choices=["live", "paper", "backtest", "daily"],
        default="paper",
        help="실행 모드 (기본값: paper)",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="CNN 가중치 α (기본 0.5)")
    parser.add_argument("--date",  type=str,   default=None,  help="--mode daily 기준일 (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.mode == "live":
        run_live(paper_trading=False, alpha=args.alpha)
    elif args.mode == "paper":
        run_live(paper_trading=True, alpha=args.alpha)
    elif args.mode == "backtest":
        run_backtest_pipeline()
    elif args.mode == "daily":
        run_daily_once(date=args.date, paper_trading=True, alpha=args.alpha)


if __name__ == "__main__":
    main()
