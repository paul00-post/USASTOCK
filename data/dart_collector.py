"""
SEC EDGAR XBRL 재무제표 수집기 (파일명은 국내 프로젝트와의 호환을 위해 유지 —
agents/agent_b.py, data/build_factor_dataset.py, main.py가 이 파일의 함수명
(load_dart_cache, get_latest_financials, run_collection)을 그대로 import하고
있어서, 내용은 전면 SEC 기반으로 바뀌었지만 파일명·함수명은 안 건드렸다).

수집 대상: 10-K(연간)/10-Q(분기) — 8-K 잠정실적 발표는 추적 안 함(2026-07-16 확정).
분기·연간 모두 "이미 처음부터 정식 재무제표"라 국내처럼 45일 확정공시 대기가
따로 필요 없다(SEC는 10-Q/10-K 제출 자체가 확정 공시).

핵심 설계 — publish_date(=filed) 기준 룩어헤드 방지는 국내와 동일 원칙:
  SEC가 같은 회계기간(예: 2020-09-30 마감 분기)의 수치를 여러 번(원 제출 +
  이후 비교공시 + 정정신고서)에 걸쳐 반복해서 태깅하는데, 이 중 **가장 먼저
  제출된(filed가 가장 이른) 값**만 취한다 — 나중 정정치를 쓰면 "그 당시엔
  몰랐을 정보"가 팩터에 새어 들어간다(예: Apple 2008년 스톡옵션 소급분배
  스캔들로 2010년에 2008년 자산총계가 재작성된 사례를 실제로 확인함).

XBRL 특유의 문제 — "누적치 vs 해당 분기 단독치":
  같은 개념(예: 매출)이 한 제출서류 안에 "이번 분기 단독 3개월" 값과
  "직전 몇 개 분기 비교용 누적" 값이 섞여 태깅된다. start~end 기간 길이로
  분기 단독(약 80~100일)과 연간 단독(약 350~380일)만 골라내고 그 외
  (반기 누적 등)는 버린다 — 이래야 국내 DART 버전과 동일하게 "분기 단독값"
  기준의 TTM 롤링 계산(factor_engine.compute_ttm_series)이 그대로 통한다.

알려진 한계 — 일부 회사·분기는 publish_date가 실제보다 늦게 잡힐 수 있음:
  회사에 따라 자기 10-Q에는 "이번 분기 단독 3개월" 수치를 명시적으로 안 태깅하고,
  다음 연도 10-K의 "분기별 요약" 각주에서야 그 수치가 처음 등장하는 경우가 있다
  (2026-07-16 Celgene 실제 데이터로 확인 — 2018년 분기별 수치가 전부 2019-02-26
  10-K에서만 나타남). 이 경우 publish_date가 실제 공개 시점보다 몇 달~1년 정도
  늦게 잡히는데, 방향이 "늦게"이지 "빠르게"가 아니라서 룩어헤드(미래 정보 유입)
  위험은 없다 — 다만 그만큼 그 분기의 팩터가 늦게 반영되는 보수적 오차는 있다.

출력:
  data/sec_raw_cache/{ticker}.json  — companyfacts API 원본 (재생성용)
  data/sec_cache/{ticker}.parquet   — 팩터 계산용 정제 데이터 (국내 dart_cache와 동일 스키마)

실행: python -m data.dart_collector [ticker] [ticker ...]
      인자 없으면 S&P500 역사적 합집합 전체 수집
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config.settings import DATA_DIR, SEC_EDGAR_USER_AGENT, SEC_SCHEMA_VERSION
from utils.logger import get_logger

logger = get_logger(__name__)

RAW_CACHE_DIR  = DATA_DIR / "sec_raw_cache"
DART_CACHE_DIR = DATA_DIR / "dart_cache"   # 국내 코드와 동일 디렉토리명 유지(팩터 계산 쪽 참조 없음, 안전)
TICKER_MAP_PATH = DATA_DIR / "sec_company_tickers.json"

_REQUEST_DELAY = 0.15  # SEC 권장(초당 10회 이하) 대비 여유
_TICKER_MAP_TTL_DAYS = 30

# 개념(concept) 후보 목록 — 회사·시대별로 태깅 관행이 달라(예: ASC606 전후로
# 매출 태그가 바뀜) 여러 후보를 순서대로 병합한다(먼저 매칭된 게 아니라
# "그 기간에 값이 있는 후보를 전부 모아 합침" — DART의 계정명 후보 매칭과
# 같은 발상).
_INSTANT_CONCEPTS: dict[str, list[str]] = {
    "total_assets":        ["Assets"],
    "total_liabilities":   ["Liabilities"],
    "total_equity":        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "current_assets":      ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash":                ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
}
_DEBT_CONCEPTS = ["LongTermDebtNoncurrent", "LongTermDebtCurrent", "LongTermDebt", "ShortTermBorrowings", "DebtCurrent"]

_FLOW_CONCEPTS: dict[str, list[str]] = {
    "revenue":          ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "op_profit":        ["OperatingIncomeLoss"],
    "net_income":       ["NetIncomeLoss"],
    "tax_expense":      ["IncomeTaxExpenseBenefit"],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt"],
    "gross_profit":     ["GrossProfit"],
    "rd_expense":       ["ResearchAndDevelopmentExpense"],
    "cfo":              ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
}

_QUARTER_DAYS_MIN, _QUARTER_DAYS_MAX = 80, 100
_ANNUAL_DAYS_MIN,  _ANNUAL_DAYS_MAX  = 350, 380


def _headers() -> dict:
    if not SEC_EDGAR_USER_AGENT:
        raise RuntimeError(
            "SEC_EDGAR_USER_AGENT 미설정 — .env에 실명+연락처 형식으로 채우세요."
        )
    return {"User-Agent": SEC_EDGAR_USER_AGENT}


# ── ticker → CIK 매핑 ─────────────────────────────────────────────────────────

def _load_ticker_cik_map(force_refresh: bool = False) -> dict[str, str]:
    """SEC company_tickers.json 기반 ticker→CIK(10자리 zero-padded) 매핑."""
    raw = None
    if not force_refresh and TICKER_MAP_PATH.exists():
        age_days = (time.time() - TICKER_MAP_PATH.stat().st_mtime) / 86400
        if age_days < _TICKER_MAP_TTL_DAYS:
            raw = json.loads(TICKER_MAP_PATH.read_text(encoding="utf-8"))

    if raw is None:
        resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=_headers(), timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        TICKER_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        TICKER_MAP_PATH.write_text(json.dumps(raw), encoding="utf-8")

    return {v["ticker"]: f"{int(v['cik_str']):010d}" for v in raw.values()}


_CIK_LOOKUP_PATH = DATA_DIR / "sec_cik_lookup.txt"


def _normalize_company_name(name: str) -> str:
    import re
    name = name.upper()
    name = re.sub(r"/[A-Z]{2,3}/?\s*$", "", name)
    name = name.replace("&", " AND ")
    name = re.sub(r"[^\sA-Z0-9]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _load_full_cik_lookup(force_refresh: bool = False) -> dict[str, str]:
    """
    SEC의 전체 이력(상장폐지 포함) 회사명→CIK 매핑. company_tickers.json은
    "지금 현재" 활성 티커만 있어서 상장폐지된 회사는 여기서 대신 찾는다
    (2026-07-16, Celgene 같은 인수합병 종목 CIK 조회 실패로 발견).
    """
    if not force_refresh and _CIK_LOOKUP_PATH.exists():
        age_days = (time.time() - _CIK_LOOKUP_PATH.stat().st_mtime) / 86400
        if age_days < _TICKER_MAP_TTL_DAYS:
            text = _CIK_LOOKUP_PATH.read_text(encoding="utf-8", errors="ignore")
            return _parse_cik_lookup(text)

    resp = requests.get("https://www.sec.gov/Archives/edgar/cik-lookup-data.txt", headers=_headers(), timeout=60)
    resp.raise_for_status()
    _CIK_LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CIK_LOOKUP_PATH.write_text(resp.text, encoding="utf-8", errors="ignore")
    return _parse_cik_lookup(resp.text)


def _parse_cik_lookup(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, cik, _ = line.rsplit(":", 2) if line.count(":") >= 2 else (None, None, None)
        if not name or not cik:
            continue
        norm = _normalize_company_name(name)
        # 같은 이름의 첫 등장(=가장 먼저 등록된 CIK)을 우선 — 자회사·후속법인보다
        # 원 등록 법인일 가능성이 높음
        mapping.setdefault(norm, cik.zfill(10))
    return mapping


def get_cik(ticker: str, company_name_hint: str | None = None) -> str | None:
    """
    단일 종목 CIK 조회(10자리 zero-padded).

    ① 활성 티커 목록(company_tickers.json)에서 먼저 찾고, 실패하면
    ② company_name_hint(예: Tiingo가 주는 회사명)로 전체 이력 CIK 목록에서 찾는다
    — 상장폐지된 종목은 활성 티커 목록에 없기 때문(2026-07-16 확인).
    """
    cik = _load_ticker_cik_map().get(ticker)
    if cik:
        return cik
    if company_name_hint:
        return _load_full_cik_lookup().get(_normalize_company_name(company_name_hint))
    return None


# ── companyfacts 원본 조회(캐시) ───────────────────────────────────────────────

def fetch_companyfacts(cik: str, ticker: str) -> dict | None:
    """SEC XBRL companyfacts API 원본 조회 — 로컬 캐시 우선."""
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_CACHE_DIR / f"{ticker}.json"

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("schema_version") == SEC_SCHEMA_VERSION:
                return cached.get("data")
        except Exception:
            pass

    time.sleep(_REQUEST_DELAY)
    try:
        resp = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers=_headers(), timeout=30,
        )
        if resp.status_code != 200:
            logger.warning("%s companyfacts 조회 실패: HTTP %s", ticker, resp.status_code)
            return None
        data = resp.json()
    except Exception as e:
        logger.error("%s companyfacts 조회 오류: %s", ticker, e)
        return None

    cache_path.write_text(
        json.dumps({"schema_version": SEC_SCHEMA_VERSION, "data": data}), encoding="utf-8"
    )
    return data


# ── XBRL 개념 추출 ────────────────────────────────────────────────────────────

def _facts_for(companyfacts: dict, concept: str) -> list[dict]:
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    units = gaap.get(concept, {}).get("units", {})
    return units.get("USD", [])


def _extract_instant(companyfacts: dict, candidates: list[str]) -> pd.DataFrame:
    """시점(Balance Sheet) 개념 — end 기준 최초 제출값(=filed 최솟값)만 채택."""
    rows = []
    for concept in candidates:
        for f in _facts_for(companyfacts, concept):
            if f.get("form") not in ("10-K", "10-Q") or not f.get("end") or f.get("val") is None:
                continue
            rows.append({"end": f["end"], "val": f["val"], "filed": f["filed"]})
    if not rows:
        return pd.DataFrame(columns=["end", "val", "filed"])
    df = pd.DataFrame(rows).sort_values("filed")
    return df.drop_duplicates(subset=["end"], keep="first")


def _extract_debt(companyfacts: dict) -> pd.DataFrame:
    """차입금 관련 개념 전부 합산 — end별로 그룹 후 각 개념의 최초 제출값을 더함."""
    parts = []
    for concept in _DEBT_CONCEPTS:
        df = _extract_instant(companyfacts, [concept])
        if not df.empty:
            parts.append(df.rename(columns={"val": concept}).drop(columns="filed"))
    if not parts:
        return pd.DataFrame(columns=["end", "val"])
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p, on="end", how="outer")
    value_cols = [c for c in merged.columns if c != "end"]
    merged["val"] = merged[value_cols].sum(axis=1, skipna=True)
    return merged[["end", "val"]]


def _extract_flow(companyfacts: dict, candidates: list[str]) -> pd.DataFrame:
    """
    유량(손익계산서·현금흐름표) 개념 — start~end 기간 길이로 "분기 단독"(80~100일)
    vs "연간 단독"(350~380일)만 골라내고, 그 외(반기누적 등)는 버린다.
    각 (end, period_type) 조합에서 최초 제출값(filed 최솟값)만 채택.

    fp(SEC가 이미 그 회사 자체 회계연도 기준으로 계산해둔 회계분기 라벨)를
    같이 들고 온다 — Apple처럼 회계연도가 9월에 끝나는 회사는 달력 12월이
    회계상 Q4가 아니라 Q1이라, 달력 월(月)로 분기를 추정하면 틀린다
    (2026-07-16 Apple 실제 데이터로 확인한 버그 — 12월 마감 분기가 계속
    "Q4"로 잘못 찍혀서 같은 라벨이 1년에 두 번 나옴).
    """
    rows = []
    for concept in candidates:
        for f in _facts_for(companyfacts, concept):
            if f.get("form") not in ("10-K", "10-Q") or not f.get("start") or not f.get("end") or f.get("val") is None:
                continue
            days = (pd.Timestamp(f["end"]) - pd.Timestamp(f["start"])).days
            if _QUARTER_DAYS_MIN <= days <= _QUARTER_DAYS_MAX:
                period_type = "Q"
            elif _ANNUAL_DAYS_MIN <= days <= _ANNUAL_DAYS_MAX:
                period_type = "FY"
            else:
                continue
            rows.append({
                "end": f["end"], "period_type": period_type, "val": f["val"],
                "filed": f["filed"], "fp": f.get("fp"),
            })
    if not rows:
        return pd.DataFrame(columns=["end", "period_type", "val", "filed", "fp"])
    df = pd.DataFrame(rows).sort_values("filed")
    return df.drop_duplicates(subset=["end", "period_type"], keep="first")


def _quarter_label(period_type: str, fp: str | None, end: str) -> str:
    """
    period_type("Q"/"FY") + fp(SEC가 이미 계산해둔 회사 자체 회계연도 기준
    분기 라벨) → Q1~Q4. FY(연간총액)는 DART 관례대로 항상 Q4로 취급.
    fp가 없거나 예상 밖 값이면 달력 월(月) 기준으로 폴백 — 단, Apple처럼
    회계연도가 1월이 아닌 회사는 이 폴백 자체가 부정확할 수 있어(fp가
    거의 항상 있으므로) 최후 수단으로만 쓴다.
    """
    if period_type == "FY":
        return "Q4"
    if fp in ("Q1", "Q2", "Q3", "Q4"):
        return fp
    month = pd.Timestamp(end).month
    return {1: "Q1", 2: "Q1", 3: "Q1", 4: "Q2", 5: "Q2", 6: "Q2",
            7: "Q3", 8: "Q3", 9: "Q3", 10: "Q4", 11: "Q4", 12: "Q4"}[month]


def _build_financials_df(companyfacts: dict, ticker: str) -> pd.DataFrame | None:
    # net_income을 "이 회사가 어떤 기간들을 보고했는지"의 기준(anchor)으로 삼는다
    # — 거의 모든 회사가 분기·연간 순이익은 빠짐없이 태깅하므로 매출 태그
    # 파편화(개념명이 시대별로 바뀌는 문제)보다 신뢰도가 높다.
    anchor = _extract_flow(companyfacts, _FLOW_CONCEPTS["net_income"])
    if anchor.empty:
        return None

    rows: list[dict[str, Any]] = []
    for _, a in anchor.iterrows():
        end, period_type, filed = a["end"], a["period_type"], a["filed"]
        row: dict[str, Any] = {
            "report_date":  end,
            "bsns_year":    pd.Timestamp(end).year,
            "quarter":      _quarter_label(period_type, a.get("fp"), end),
            "publish_date": filed,
            "net_income":   a["val"],
        }
        for factor, candidates in _FLOW_CONCEPTS.items():
            if factor == "net_income":
                continue
            df = _extract_flow(companyfacts, candidates)
            match = df[(df["end"] == end) & (df["period_type"] == period_type)]
            row[factor] = float(match.iloc[0]["val"]) if not match.empty else None
        for factor, candidates in _INSTANT_CONCEPTS.items():
            df = _extract_instant(companyfacts, candidates)
            match = df[df["end"] == end]
            row[factor] = float(match.iloc[0]["val"]) if not match.empty else None
        debt_df = _extract_debt(companyfacts)
        match = debt_df[debt_df["end"] == end]
        row["total_debt"] = float(match.iloc[0]["val"]) if not match.empty else None
        rows.append(row)

    df = pd.DataFrame(rows)
    df["ticker"] = ticker
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df["report_date"]  = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.sort_values("report_date").reset_index(drop=True)
    return df


# ── 공개 인터페이스 ───────────────────────────────────────────────────────────

_TIINGO_DELAY = 75.0  # data/build_price_cache.py와 동일 — 시간당 50회 실제 제한 준수


def _tiingo_name_hint(ticker: str) -> str | None:
    """활성 티커 목록에 없는(상장폐지) 종목의 CIK를 찾기 위해 Tiingo에서 회사명만 빌려온다."""
    from config.settings import TIINGO_API_KEY
    if not TIINGO_API_KEY:
        return None
    try:
        time.sleep(_TIINGO_DELAY)
        r = requests.get(
            f"https://api.tiingo.com/tiingo/daily/{ticker}",
            params={"token": TIINGO_API_KEY}, timeout=15,
        )
        if r.status_code != 200:
            return None
        return r.json().get("name")
    except Exception:
        return None


def get_cik_with_fallback(ticker: str) -> str | None:
    """
    get_cik()에 "활성 티커 목록에 없으면 Tiingo 회사명으로 재시도" 폴백을 더한
    버전 — 상장폐지 종목(Celgene 등)은 활성 목록에 없어 이 폴백이 필요하다
    (2026-07-16 발견). collect_ticker·fetch_shares.py 양쪽에서 공용으로 쓴다.
    """
    cik = get_cik(ticker)
    if cik:
        return cik
    name_hint = _tiingo_name_hint(ticker)
    return get_cik(ticker, company_name_hint=name_hint) if name_hint else None


def collect_ticker(ticker: str, cik: str | None = None) -> pd.DataFrame | None:
    """단일 종목 전 기간 재무데이터 수집."""
    DART_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cik = cik or get_cik_with_fallback(ticker)
    if cik is None:
        logger.warning("%s: CIK 없음 — 스킵", ticker)
        return None

    companyfacts = fetch_companyfacts(cik, ticker)
    if companyfacts is None:
        return None

    df = _build_financials_df(companyfacts, ticker)
    if df is None or df.empty:
        logger.warning("%s: 수집된 재무데이터 없음", ticker)
        return None

    cache_path = DART_CACHE_DIR / f"{ticker}.parquet"
    df.to_parquet(cache_path, index=False)
    logger.info("%s: %d 분기 저장 → %s", ticker, len(df), cache_path.name)
    return df


def load_dart_cache(ticker: str) -> pd.DataFrame | None:
    """캐시된 parquet 로드. 없으면 None."""
    path = DART_CACHE_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def get_latest_financials(ticker: str, ref_date: str | pd.Timestamp) -> pd.Series | None:
    """
    ref_date 기준 가장 최근 확정 공시 재무데이터 반환.
    룩어헤드 방지: publish_date <= ref_date 필터 적용.
    """
    df = load_dart_cache(ticker)
    if df is None or df.empty:
        return None
    ref = pd.Timestamp(ref_date)
    valid = df[df["publish_date"].notna() & (df["publish_date"] <= ref)]
    if valid.empty:
        return None
    return valid.sort_values("publish_date").iloc[-1]


def run_collection(tickers: list[str] | None = None) -> None:
    """
    지정 종목(None이면 S&P500 역사적 합집합) SEC 재무 데이터 수집.
    과거에 편입됐다가 빠진 종목도 포함하여 백테스팅 생존편향을 방지.
    """
    if tickers is None:
        from data.build_universe import get_all_tickers_until
        tickers = get_all_tickers_until(datetime.today().year)
        logger.info("S&P500 역사적 합집합 전체 수집: %d 종목", len(tickers))

    failed = []
    for i, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] %s 수집 중...", i, len(tickers), ticker)
        try:
            collect_ticker(ticker)
        except Exception as e:
            logger.error("%s 수집 실패: %s", ticker, e)
            failed.append(ticker)

    if failed:
        logger.warning("수집 실패 종목 (%d개): %s", len(failed), failed)
    logger.info("SEC 재무데이터 수집 완료")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    run_collection(tickers=args if args else None)
