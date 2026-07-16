"""
한국투자증권(KIS) Open API 클라이언트 — REST 기반 주문 전송.

키움(OCX 방식)과 달리 순수 HTTP 호출이라 Windows 로그인 세션 없이도 동작한다.

필요한 .env 값 (config.settings.KIS_CONFIG 참고):
  KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO, KIS_ACCOUNT_PRODUCT_CD, KIS_MOCK

사용 전 준비물:
  1. KIS 개발자센터(https://apiportal.koreainvestment.com)에서 앱 등록 → APP_KEY/APP_SECRET 발급
  2. 모의투자 계좌로 먼저 테스트 (KIS_MOCK=true) 후 실전 전환 (KIS_MOCK=false)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from config.settings import BASE_DIR, KIS_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

_REAL_BASE = "https://openapi.koreainvestment.com:9443"
_MOCK_BASE = "https://openapivts.koreainvestment.com:29443"

# tr_id: (실전, 모의) — 국내주식 현금 매수/매도 주문
_TR_ID = {
    "buy":  {"real": "TTTC0802U", "mock": "VTTC0802U"},
    "sell": {"real": "TTTC0801U", "mock": "VTTC0801U"},
}


def _base_url() -> str:
    return _MOCK_BASE if KIS_CONFIG["mock"] else _REAL_BASE


def _token_cache_path() -> Path:
    # 모의/실전 토큰은 서로 다른 서버에서 발급되어 호환되지 않는데, 예전엔 캐시 파일을
    # 하나만 써서 KIS_MOCK을 바꿔도 이전 모드의 토큰을 그대로 재사용하는 버그가 있었다
    # (2026-07-11 실전 전환 중 "기간이 만료된 token" EGW00123 오류로 발견 — 실제로는
    # 만료가 아니라 모의투자 토큰을 실전 서버에 쓴 것).
    suffix = "mock" if KIS_CONFIG["mock"] else "real"
    return BASE_DIR / "data" / f"kis_token_cache_{suffix}.json"


def _load_cached_token() -> str | None:
    path = _token_cache_path()
    if not path.exists():
        return None
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
        # 만료 60초 전에는 재발급 (여유 버퍼)
        if cache.get("expires_at", 0) > time.time() + 60:
            return cache["access_token"]
    except Exception:
        pass
    return None


def _save_token_cache(access_token: str, expires_in: int) -> None:
    path = _token_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"access_token": access_token, "expires_at": time.time() + expires_in}),
        encoding="utf-8",
    )


def get_access_token() -> str | None:
    """
    OAuth 접근토큰 발급 (24시간 유효, 파일 캐싱으로 재발급 최소화).
    KIS는 토큰 재발급 요청을 과도하게 하면 1분당 요청 제한에 걸릴 수 있음.
    """
    cached = _load_cached_token()
    if cached:
        return cached

    if not KIS_CONFIG["app_key"] or not KIS_CONFIG["app_secret"]:
        logger.warning("KIS_APP_KEY/KIS_APP_SECRET 미설정 — 토큰 발급 불가")
        return None

    try:
        resp = requests.post(
            f"{_base_url()}/oauth2/tokenP",
            headers={"content-type": "application/json"},
            json={
                "grant_type": "client_credentials",
                "appkey":     KIS_CONFIG["app_key"],
                "appsecret":  KIS_CONFIG["app_secret"],
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        _save_token_cache(data["access_token"], int(data.get("expires_in", 86400)))
        return data["access_token"]
    except Exception as e:
        logger.error("KIS 토큰 발급 실패: %s", e)
        return None


def _get_hashkey(body: dict) -> str | None:
    """주문 요청 위변조 방지용 해시 (KIS 주문 API 필수 헤더)."""
    try:
        resp = requests.post(
            f"{_base_url()}/uapi/hashkey",
            headers={
                "content-type": "application/json",
                "appkey":       KIS_CONFIG["app_key"],
                "appsecret":    KIS_CONFIG["app_secret"],
            },
            json=body,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["HASH"]
    except Exception as e:
        logger.error("KIS 해시키 발급 실패: %s", e)
        return None


def _submit_order(action: str, ticker: str, shares: int, price: int, retries: int = 2) -> dict | None:
    """
    국내주식 현금 매수/매도 주문 (지정가) 실제 전송.

    모의투자 서버는 초당 5건 제한이 있어 순간적으로 500을 반환하는 경우가 있다
    (다른 조회 함수들과 동일하게 짧게 재시도한다).

    반환값은 "주문이 접수됐다"는 뜻이지 "실제로 체결됐다"는 뜻이 아니다 —
    지정가 주문은 접수만 되고 시장에서 아직 상대방을 못 만나 미체결로 남을 수 있다.
    체결 여부까지 확인하려면 send_order_verified()를 쓴다.
    """
    token = get_access_token()
    if token is None:
        logger.warning("KIS 접근토큰 없음 — 주문 전송 불가 (페이퍼트레이딩 모드로 대체)")
        return None

    account_no  = KIS_CONFIG["account_no"]
    product_cd  = KIS_CONFIG["account_product_cd"]
    if not account_no:
        logger.warning("KIS_ACCOUNT_NO 미설정 — 주문 전송 불가")
        return None

    body = {
        "CANO":         account_no,
        "ACNT_PRDT_CD": product_cd,
        "PDNO":         ticker,
        "ORD_DVSN":     "00",          # 00 = 지정가
        "ORD_QTY":      str(shares),
        "ORD_UNPR":     str(price),
    }

    tr_id = _TR_ID[action]["mock" if KIS_CONFIG["mock"] else "real"]

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        hashkey = _get_hashkey(body)
        if hashkey is None:
            last_err = RuntimeError("해시키 발급 실패")
            if attempt < retries:
                time.sleep(0.3 * (attempt + 1))
                continue
            break

        try:
            resp = requests.post(
                f"{_base_url()}/uapi/domestic-stock/v1/trading/order-cash",
                headers={
                    "content-type":  "application/json; charset=utf-8",
                    "authorization": f"Bearer {token}",
                    "appkey":        KIS_CONFIG["app_key"],
                    "appsecret":     KIS_CONFIG["app_secret"],
                    "tr_id":         tr_id,
                    "custtype":      "P",
                    "hashkey":       hashkey,
                },
                json=body,
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            ok = result.get("rt_cd") == "0"
            if ok:
                logger.info("KIS %s 주문 접수 성공: %s %d주 @%d — %s", action, ticker, shares, price, result.get("msg1"))
                output = result.get("output", {}) or {}
                return {
                    "odno":              output.get("ODNO"),
                    "krx_fwdg_ord_orgno": output.get("KRX_FWDG_ORD_ORGNO"),
                }
            else:
                logger.error("KIS %s 주문 실패: %s %d주 @%d — %s", action, ticker, shares, price, result.get("msg1"))
            return None
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.3 * (attempt + 1))
                continue

    logger.error("KIS %s 주문 요청 오류 %s (재시도 %d회 소진): %s", action, ticker, retries, last_err)
    return None


def send_order(action: str, ticker: str, shares: int, price: int, retries: int = 2) -> bool:
    """기존 호출부와의 호환용 — 주문 접수 성공 여부만 반환(체결 확인 없음)."""
    return _submit_order(action, ticker, shares, price, retries) is not None


_FILL_TR_ID   = {"real": "TTTC0081R", "mock": "VTTC0081R"}
_CANCEL_TR_ID = {"real": "TTTC0013U", "mock": "VTTC0013U"}


def get_order_fill_status(ticker: str, odno: str) -> dict | None:
    """
    당일 특정 주문번호(odno)의 체결 수량 조회.

    Returns
    -------
    dict: {"filled_qty": int, "ord_gno_brno": str} 또는 조회 실패 시 None
    """
    token = get_access_token()
    if token is None or not odno:
        return None

    today = pd.Timestamp.today().strftime("%Y%m%d")
    tr_id = _FILL_TR_ID["mock" if KIS_CONFIG["mock"] else "real"]

    try:
        resp = requests.get(
            f"{_base_url()}/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
            headers={
                "content-type":  "application/json; charset=utf-8",
                "authorization": f"Bearer {token}",
                "appkey":        KIS_CONFIG["app_key"],
                "appsecret":     KIS_CONFIG["app_secret"],
                "tr_id":         tr_id,
                "custtype":      "P",
            },
            params={
                "CANO": KIS_CONFIG["account_no"], "ACNT_PRDT_CD": KIS_CONFIG["account_product_cd"],
                "INQR_STRT_DT": today, "INQR_END_DT": today,
                "SLL_BUY_DVSN_CD": "00", "PDNO": ticker,
                "CCLD_DVSN": "00", "INQR_DVSN": "00", "INQR_DVSN_3": "00",
                "ORD_GNO_BRNO": "", "ODNO": odno, "INQR_DVSN_1": "",
                "CTX_AREA_FK100": "", "CTX_AREA_NK100": "",
                "EXCG_ID_DVSN_CD": "KRX",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("rt_cd") != "0":
            logger.error("KIS 체결 조회 실패: %s", data.get("msg1"))
            return None
        row = next((r for r in data.get("output1", []) if r.get("odno") == odno), None)
        if row is None:
            return None
        avg_price = float(row.get("avg_prvs", 0) or 0)
        return {
            "filled_qty":   int(float(row.get("tot_ccld_qty", 0) or 0)),
            "avg_price":    avg_price if avg_price > 0 else None,
            "ord_gno_brno": row.get("ord_gno_brno"),
        }
    except Exception as e:
        logger.error("KIS 체결 조회 오류 %s (odno=%s): %s", ticker, odno, e)
        return None


def cancel_order(ord_gno_brno: str, odno: str, qty: int, price: int) -> bool:
    """미체결(또는 부분체결) 지정가 주문 취소."""
    token = get_access_token()
    if token is None or not ord_gno_brno or not odno:
        return False

    body = {
        "CANO":               KIS_CONFIG["account_no"],
        "ACNT_PRDT_CD":       KIS_CONFIG["account_product_cd"],
        "KRX_FWDG_ORD_ORGNO": ord_gno_brno,
        "ORGN_ODNO":          odno,
        "ORD_DVSN":           "00",
        "RVSE_CNCL_DVSN_CD":  "02",  # 02 = 취소
        "ORD_QTY":            str(qty),
        "ORD_UNPR":           str(price),
        "QTY_ALL_ORD_YN":     "Y",
        "EXCG_ID_DVSN_CD":    "KRX",
    }
    hashkey = _get_hashkey(body)
    if hashkey is None:
        return False

    tr_id = _CANCEL_TR_ID["mock" if KIS_CONFIG["mock"] else "real"]
    try:
        resp = requests.post(
            f"{_base_url()}/uapi/domestic-stock/v1/trading/order-rvsecncl",
            headers={
                "content-type":  "application/json; charset=utf-8",
                "authorization": f"Bearer {token}",
                "appkey":        KIS_CONFIG["app_key"],
                "appsecret":     KIS_CONFIG["app_secret"],
                "tr_id":         tr_id,
                "custtype":      "P",
                "hashkey":       hashkey,
            },
            json=body,
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()
        ok = result.get("rt_cd") == "0"
        if ok:
            logger.info("주문 취소 성공: odno=%s", odno)
        else:
            logger.error("주문 취소 실패: odno=%s — %s", odno, result.get("msg1"))
        return ok
    except Exception as e:
        logger.error("주문 취소 오류 odno=%s: %s", odno, e)
        return False


def send_order_fill(action: str, ticker: str, shares: int, price: int, wait_sec: int = 4) -> dict:
    """
    주문을 접수하고, 실제로 얼마나 체결됐는지까지 확인한다.

    지정가 주문은 접수(rt_cd=0)돼도 시장에서 상대방을 못 만나면 그냥 미체결로
    남는다 — 이걸 "체결 완료"로 착각해서 장부(cash/holdings)를 먼저 바꿔버리면
    실제 계좌와 어긋난다(2026-07 대우건설 건에서 실제로 발생). 접수 후 wait_sec초
    기다렸다가 체결 수량을 재조회하고, 미체결로 남은 수량은 주문을 취소해
    호출부가 그 잔여분만 새 가격으로 재시도할 수 있게 한다.

    Returns
    -------
    dict: {"filled_qty": int, "avg_price": float | None}
          filled_qty < shares 이면 나머지는 취소된 상태 (재주문은 호출부 책임)
    """
    submitted = _submit_order(action, ticker, shares, price)
    if submitted is None:
        return {"filled_qty": 0, "avg_price": None}

    odno = submitted.get("odno")
    if not odno:
        logger.warning("%s %s 주문번호를 받지 못해 체결 확인 불가 — 접수 성공으로 간주", action, ticker)
        return {"filled_qty": shares, "avg_price": float(price)}

    time.sleep(wait_sec)
    status = get_order_fill_status(ticker, odno)
    if status is None:
        logger.warning("%s %s 체결 조회 실패 — 접수 성공으로 간주(다음 동기화에서 확인됨)", action, ticker)
        return {"filled_qty": shares, "avg_price": float(price)}

    filled = status["filled_qty"]
    if filled >= shares:
        logger.info("%s %s 체결 확인 완료: %d/%d주", action, ticker, filled, shares)
        return {"filled_qty": filled, "avg_price": status.get("avg_price") or float(price)}

    logger.warning(
        "%s %s 체결 확인 실패(%d/%d주만 체결) — 미체결분 취소 시도, 호출부에서 잔여분 재시도",
        action, ticker, filled, shares,
    )
    cancel_ok = cancel_order(status.get("ord_gno_brno"), odno, shares, price)

    # 취소 요청과 거래소 체결이 경합(race)할 수 있다 — 취소가 "성공"으로 와도
    # 그 사이에 실제로는 체결이 끝났을 수 있어(2026-07-09 대우건설·대한조선 건에서
    # 실제로 발생: cncl_yn=N인데 우리는 취소된 줄 알고 장부에서 놓침), 취소 시도
    # 직후 체결 수량을 한 번 더 재확인해 최종 상태를 확정한다.
    time.sleep(2)
    final_status = get_order_fill_status(ticker, odno)
    final_filled = final_status["filled_qty"] if final_status else filled
    final_avg    = (final_status.get("avg_price") if final_status else None) or status.get("avg_price")

    if final_filled > filled:
        logger.warning(
            "%s %s 취소 이후 재확인 결과 추가 체결 확인됨(%d→%d/%d주, 취소응답=%s) — 취소가 체결과 경합했을 가능성",
            action, ticker, filled, final_filled, shares, cancel_ok,
        )
    if final_filled > 0:
        logger.warning("%s %s 최종 체결 %d주 확정 (평균단가=%s)", action, ticker, final_filled, final_avg)
    return {"filled_qty": final_filled, "avg_price": final_avg if final_filled > 0 else None}


def send_order_verified(action: str, ticker: str, shares: int, price: int, wait_sec: int = 4) -> bool:
    """기존 호출부와의 호환용 — 요청 수량 전체가 체결된 경우에만 True."""
    return send_order_fill(action, ticker, shares, price, wait_sec)["filled_qty"] >= shares


# ── 계좌 잔고 조회 (실제 잔고와 portfolio_state.json 동기화용) ────────────────

_BALANCE_TR_ID = {"real": "TTTC8434R", "mock": "VTTC8434R"}


def get_account_balance(retries: int = 2) -> dict | None:
    """
    실제 계좌 예수금·보유종목 조회. portfolio_state.json은 시스템이 자체적으로
    계산해 나가는 장부라 실제 계좌와 어긋날 수 있다 — 이 함수로 진짜 잔고를
    확인해 동기화한다.

    Returns
    -------
    dict: {"cash": float, "holdings": {ticker: shares}} 또는 None
    """
    token = get_access_token()
    if token is None:
        return None

    account_no = KIS_CONFIG["account_no"]
    product_cd = KIS_CONFIG["account_product_cd"]
    if not account_no:
        logger.warning("계좌번호 미설정 — 잔고 조회 불가")
        return None

    tr_id = _BALANCE_TR_ID["mock" if KIS_CONFIG["mock"] else "real"]

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                f"{_base_url()}/uapi/domestic-stock/v1/trading/inquire-balance",
                headers={
                    "content-type":  "application/json; charset=utf-8",
                    "authorization": f"Bearer {token}",
                    "appkey":        KIS_CONFIG["app_key"],
                    "appsecret":     KIS_CONFIG["app_secret"],
                    "tr_id":         tr_id,
                    "custtype":      "P",
                },
                params={
                    "CANO": account_no, "ACNT_PRDT_CD": product_cd,
                    "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02",
                    "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N",
                    "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "01",
                    "CTX_AREA_FK100": "", "CTX_AREA_NK100": "",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") != "0":
                logger.error("KIS 잔고 조회 실패: %s", data.get("msg1"))
                return None

            summary  = data.get("output2", [{}])
            summary  = summary[0] if summary else {}
            # dnca_tot_amt(예수금총액)는 계좌 최초 입금 총액 기준이라 매매를 해도 안 바뀐다
            # (모의투자 계좌에서 확인됨 — 항상 최초 입금액 그대로).
            # 실제 남은 현금은 prvs_rcdl_excc_amt(가수도정산금액)이며
            # "초기 입금액 - 누적 매수액 - 수수료"와 정확히 일치한다.
            cash     = float(summary.get("prvs_rcdl_excc_amt", 0) or 0)

            holdings: dict[str, int] = {}
            for row in data.get("output1", []):
                ticker = row.get("pdno")
                shares = int(float(row.get("hldg_qty", 0) or 0))
                if ticker and shares > 0:
                    holdings[ticker] = shares

            return {"cash": cash, "holdings": holdings}
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.3 * (attempt + 1))
                continue
    logger.error("KIS 잔고 조회 오류 (재시도 %d회 소진): %s", retries, last_err)
    return None


# ── 실시간 시세 조회 (장중 손절 감시 · 시가 확인용) ──────────────────────────

def get_current_price(ticker: str, retries: int = 2) -> dict | None:
    """
    주식현재가 시세 조회. 오늘 시가 확인(09:01 진입)과 현재가 확인(틱 계산)에 사용.
    모의투자 서버가 간헐적으로 500을 반환하는 경우가 있어 짧게 재시도한다.

    Returns
    -------
    dict: current, open, high, low, prev_close (모두 float) 또는 None
    """
    token = get_access_token()
    if token is None:
        return None

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                f"{_base_url()}/uapi/domestic-stock/v1/quotations/inquire-price",
                headers={
                    "content-type":  "application/json; charset=utf-8",
                    "authorization": f"Bearer {token}",
                    "appkey":        KIS_CONFIG["app_key"],
                    "appsecret":     KIS_CONFIG["app_secret"],
                    "tr_id":         "FHKST01010100",
                    "custtype":      "P",
                },
                params={
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": ticker,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") != "0":
                logger.error("KIS 현재가 조회 실패: %s — %s", ticker, data.get("msg1"))
                return None
            o = data.get("output", {})
            return {
                "current":    float(o.get("stck_prpr", 0) or 0),
                "open":       float(o.get("stck_oprc", 0) or 0),
                "high":       float(o.get("stck_hgpr", 0) or 0),
                "low":        float(o.get("stck_lwpr", 0) or 0),
                "prev_close": float(o.get("stck_sdpr", 0) or 0),
            }
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.3 * (attempt + 1))
                continue
    logger.error("KIS 현재가 조회 오류 %s (재시도 %d회 소진): %s", ticker, retries, last_err)
    return None


def get_minute_candles(ticker: str, count: int = 30, retries: int = 2) -> pd.DataFrame | None:
    """
    주식당일분봉조회 — 당일 1분봉 시가/고가/저가/종가 (최신 순).
    장중 손절 감시에서 "마지막 체크 이후 구간에 손절선을 스쳤는지" 확인할 때 사용.
    모의투자 서버가 간헐적으로 500을 반환하는 경우가 있어 짧게 재시도한다.

    Returns
    -------
    DataFrame(columns=[date, time, open, high, low, close]) 또는 None
    """
    token = get_access_token()
    if token is None:
        return None

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                f"{_base_url()}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
                headers={
                    "content-type":  "application/json; charset=utf-8",
                    "authorization": f"Bearer {token}",
                    "appkey":        KIS_CONFIG["app_key"],
                    "appsecret":     KIS_CONFIG["app_secret"],
                    "tr_id":         "FHKST03010200",
                    "custtype":      "P",
                },
                params={
                    "FID_ETC_CLS_CODE":       "",
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD":         ticker,
                    "FID_INPUT_HOUR_1":       "",   # 빈 값 = 현재 시각까지
                    "FID_PW_DATA_INCU_YN":    "Y",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("rt_cd") != "0":
                logger.error("KIS 분봉 조회 실패: %s — %s", ticker, data.get("msg1"))
                return None
            rows = data.get("output2", [])
            if not rows:
                return None
            df = pd.DataFrame(rows).rename(columns={
                "stck_bsop_date": "date", "stck_cntg_hour": "time",
                "stck_oprc": "open", "stck_hgpr": "high",
                "stck_lwpr": "low",  "stck_prpr": "close",
            })
            for c in ["open", "high", "low", "close"]:
                df[c] = df[c].astype(float)
            return df.head(count)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.3 * (attempt + 1))
                continue
    logger.error("KIS 분봉 조회 오류 %s (재시도 %d회 소진): %s", ticker, retries, last_err)
    return None
