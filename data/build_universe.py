"""
S&P500 과거 시점 구성종목 스냅샷 빌더 — Vanguard 500 Index Fund SEC 반기보고서 기반.

소스: Vanguard 500 Index Fund(CIK 0000036405, 1976년 세계 최초의 인덱스펀드)가
SEC에 제출하는 N-30D(~2003년경까지)/N-CSR(2003년경~) 반기보고서의
"Statement of Net Assets"에는 그 시점 실제 보유종목 전체가 회사명 기준으로
실려있다. 위키피디아 "List of S&P 500 companies" 방식과 달리 그 문서가
2005-09-14에야 생성된 제약이 없다(뱅가드 500은 1976년부터 존재) — 2000년부터
지금까지 연 2회(3월·8월 제출, 전년 12월말·당해 6월말 기준) 스냅샷을 균일하게
확보할 수 있다(2026-07-16 확정, 실제 1999년 필라이언 열어 Microsoft/GE/Cisco
등 정확한 당시 상위 보유종목을 확인해 검증 완료).

보고서 형식이 시기별로 다르다:
  - ~2003년경: N-30D, 고정폭 텍스트(.txt) — 회사명이 개행으로 줄바꿈되기도 함
  - 2003년경~: N-CSR, HTML — "Statement of Net Assets" 표가 여러 <table>에
    걸쳐 나뉘어 있고, 종목명이 두 번째 열에 있다(첫 열은 각주 마커).

보고서엔 티커가 아니라 회사명만 있어 SEC company_tickers.json 기준 이름
매칭이 필요하다. 25년간 사명 변경·합병으로 정확히 안 붙는 이름은
data/universe_name_overrides.csv에 수동으로 채워 넣는다(fuzzy 추정값은
data/universe_name_unmatched.csv에 참고용으로만 기록 — 잘못 매칭될 위험이
있어 자동으로 스냅샷에 반영하지 않는다).

출력: data/universe_snapshots/{YYYYMM}.parquet (ticker 컬럼 포함, 기존 계약 유지)
      YYYYMM은 보고서 기준일(회계연도 말) — 12월말 스냅샷은 YYYY12, 6월말은 YYYY06.

실행: python -m data.build_universe [start_year] [end_year]
      기본값: start_year=2000, end_year=현재연도
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config.settings import DATA_DIR, SEC_EDGAR_USER_AGENT, VANGUARD_500_CIK
from utils.logger import get_logger

logger = get_logger(__name__)

SNAPSHOT_DIR      = DATA_DIR / "universe_snapshots"
FILING_CACHE_DIR  = DATA_DIR / "vanguard_500_filings"
TICKER_MAP_PATH   = DATA_DIR / "sec_company_tickers.json"
NAME_OVERRIDE_CSV = DATA_DIR / "universe_name_overrides.csv"    # ticker,raw_name (수동 유지)
UNMATCHED_CSV     = DATA_DIR / "universe_name_unmatched.csv"    # 자동 생성 — 검토용

_REQUEST_DELAY   = 0.2   # SEC 요청 간격 (초)
_TICKER_MAP_TTL_DAYS = 30
_FUZZY_CUTOFF    = 0.88  # 참고용 제안 임계값 (자동 반영 안 함)

# N-CSR로 전환된 시점 근방 — 이 날짜 이전 필링은 고정폭 텍스트(.txt),
# 이후는 HTML로 간주(2026-07-16, 실제 필링 몇 건 열어보고 확인한 경계).
_LEGACY_TEXT_CUTOFF = pd.Timestamp("2003-01-01")


def _headers() -> dict:
    if not SEC_EDGAR_USER_AGENT:
        raise RuntimeError(
            "SEC_EDGAR_USER_AGENT 미설정 — .env에 실명+연락처 형식으로 채우세요 "
            "(예: 'MyCompany contact@example.com'). 형식 안 지키면 SEC가 요청을 차단함."
        )
    return {"User-Agent": SEC_EDGAR_USER_AGENT}


# ── 1. 필링(제출 이력) 조회 ───────────────────────────────────────────────────

def _list_filings(form_type: str) -> list[dict]:
    """CIK의 특정 폼타입 제출 이력 전체 반환 (date, accession, index_href)."""
    resp = requests.get(
        "https://www.sec.gov/cgi-bin/browse-edgar",
        params={
            "action": "getcompany", "CIK": VANGUARD_500_CIK, "type": form_type,
            "dateb": "", "owner": "include", "count": "300", "output": "atom",
        },
        headers=_headers(), timeout=30,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for entry in root.findall("a:entry", ns):
        content = entry.find("a:content", ns)
        if content is None:
            continue
        date_el = content.find("a:filing-date", ns)
        href_el = content.find("a:filing-href", ns)
        acc_el  = content.find("a:accession-number", ns)
        if date_el is None or href_el is None:
            continue
        out.append({
            "date": date_el.text,
            "index_href": href_el.text,
            "accession": acc_el.text if acc_el is not None else None,
        })
    return out


def _get_all_filings() -> list[dict]:
    """N-30D + N-CSR 전체 제출 이력, 접수일 오름차순, 중복 제거."""
    filings = _list_filings("N-30D") + _list_filings("N-CSR")
    seen: set[str] = set()
    out = []
    for f in sorted(filings, key=lambda f: f["date"]):
        if f["accession"] in seen:
            continue
        seen.add(f["accession"])
        out.append(f)
    return out


# ── 2. 필링 원문 다운로드 (캐시) ───────────────────────────────────────────────

def _primary_document_url(filing: dict) -> tuple[str, bool]:
    """
    필링의 본문 문서 URL과 "레거시 텍스트인지" 여부 반환.

    2003년 이전(N-30D)은 .txt 제출물 자체가 문서 전체다.
    이후(N-CSR)는 여러 파일이 같이 묶여있어 index.json에서 가장 큰 .htm
    파일(실제 재무제표 본문)을 고른다 — 인증서(cert302.htm 등)는 훨씬 작다.
    """
    is_legacy = pd.Timestamp(filing["date"]) < _LEGACY_TEXT_CUTOFF
    accession_nodash = filing["accession"].replace("-", "")
    cik_int = str(int(VANGUARD_500_CIK))

    if is_legacy:
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{filing['accession']}.txt"
        return url, True

    time.sleep(_REQUEST_DELAY)
    idx_resp = requests.get(
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/index.json",
        headers=_headers(), timeout=30,
    )
    idx_resp.raise_for_status()
    items = idx_resp.json()["directory"]["item"]
    htm_items = [it for it in items if it["name"].lower().endswith((".htm", ".html"))]
    if not htm_items:
        raise ValueError(f"{filing['accession']}: htm 문서 없음")
    biggest = max(htm_items, key=lambda it: int(it.get("size") or 0))
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{biggest['name']}"
    return url, False


def _fetch_filing_document(filing: dict) -> tuple[str, bool]:
    """필링 본문을 캐시에서 로드하거나 다운로드. (텍스트, is_legacy) 반환."""
    FILING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FILING_CACHE_DIR / f"{filing['accession']}.cache"

    if cache_path.exists():
        meta_path = cache_path.with_suffix(".meta.json")
        is_legacy = json.loads(meta_path.read_text(encoding="utf-8"))["is_legacy"] if meta_path.exists() else True
        return cache_path.read_text(encoding="utf-8", errors="ignore"), is_legacy

    url, is_legacy = _primary_document_url(filing)
    time.sleep(_REQUEST_DELAY)
    resp = requests.get(url, headers=_headers(), timeout=60)
    resp.raise_for_status()
    text = resp.text

    cache_path.write_text(text, encoding="utf-8")
    cache_path.with_suffix(".meta.json").write_text(
        json.dumps({"is_legacy": is_legacy, "url": url}), encoding="utf-8"
    )
    return text, is_legacy


# ── 3-A. 레거시 고정폭 텍스트(N-30D, ~2003년 이전) 파싱 ───────────────────────

_ROW_RE = re.compile(r"^[\s\-]{0,4}(?P<name>.+?)\s{2,}(?P<shares>[\d,]+)\s+\$?\s*(?P<value>[\d,]+)\s*$")


def _parse_legacy_text(text: str) -> list[str]:
    """
    N-30D 고정폭 텍스트에서 500 Index Fund "Common Stocks" 섹션의 회사명 목록 추출.

    문서에 500/Growth/Value/Total Stock Market 4개 펀드가 순서대로 실려있고
    500 Index Fund가 항상 첫 번째라, 문서 전체에서 첫 "COMMON STOCKS (" ~
    첫 "TOTAL COMMON STOCKS" 구간만 보면 된다.
    """
    lines = text.splitlines()

    start_idx = None
    for i, l in enumerate(lines):
        if "COMMON STOCKS (" in l.upper():
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError("COMMON STOCKS 섹션 시작을 못 찾음")

    stop_idx = None
    for j in range(start_idx, len(lines)):
        if "TOTAL COMMON STOCKS" in lines[j].upper():
            stop_idx = j
            break
    if stop_idx is None:
        raise ValueError("TOTAL COMMON STOCKS(섹션 끝)를 못 찾음")

    names: list[str] = []
    pending: str | None = None
    for raw in lines[start_idx:stop_idx]:
        line = raw.rstrip("\n")
        stripped = line.strip()
        # 구분선("- ---...") · 빈 줄 · SGML 테이블 태그(<S> <C> <C>)는 스킵
        if not stripped or set(stripped.replace(" ", "")) <= {"-"} or stripped.startswith("<"):
            continue
        m = _ROW_RE.match(line)
        if m:
            name = m.group("name").strip()
            if pending:
                name = f"{pending} {name}"
                pending = None
            names.append(name)
        else:
            # 주수·평가액 없이 이름만 있는 줄 → 다음 줄에 이어지는 회사명 앞부분
            pending = f"{pending} {stripped}" if pending else stripped

    return names


# ── 3-B. 현대 HTML(N-CSR, 2003년경~) 파싱 ─────────────────────────────────────

def _parse_modern_html(html: str) -> list[str]:
    """
    N-CSR HTML에서 500 Index Fund "Common Stocks" 표의 회사명 목록 추출.

    이 표는 pandas.read_html로 한 번에 못 읽는다 — Vanguard가 페이지 단위로
    <table>을 여러 개로 쪼개놓기 때문. "Common Stocks (" 텍스트가 있는 표를
    시작점으로 잡고, 이후 모든 <table>을 순서대로 훑으며 "Total Common
    Stocks" 행이 나올 때까지 4-셀 행([마커, 회사명, 주수, 평가액])만 수집한다.
    회사명이 길어 두 행에 걸쳐 나뉘는 경우(주수·평가액 칸이 비어있는 행)는
    다음 행과 합친다.
    """
    soup = BeautifulSoup(html, "html.parser")
    start_node = soup.find(string=re.compile(r"Common Stocks \("))
    if start_node is None:
        raise ValueError("Common Stocks 섹션 시작을 못 찾음")
    start_table = start_node.find_parent("table")
    if start_table is None:
        raise ValueError("Common Stocks 표를 못 찾음")

    names: list[str] = []
    pending: str | None = None
    for tbl in [start_table] + start_table.find_all_next("table"):
        stop = False
        for tr in tbl.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            texts = [c.get_text(strip=True) for c in cells]
            if texts and "Total Common Stocks" in texts[0]:
                stop = True
                break
            if len(texts) != 4:
                continue
            name_cell = texts[1]
            if not name_cell:
                continue
            if pending:
                name_cell = f"{pending} {name_cell}"
                pending = None
            if not texts[2] and not texts[3]:
                # 주수·평가액이 비어있음 → 회사명이 다음 행에 이어짐
                pending = name_cell
                continue
            names.append(name_cell)
        if stop:
            break

    return names


def _fund_holdings_or_none(filing: dict) -> list[str] | None:
    """
    필링 본문에 "500 Index Fund" 텍스트가 없으면 이 accession은 500 펀드가 아닌
    형제 펀드(Extended Market/Mid-Cap x3/Small-Cap x3/Total Stock Market 등)
    묶음 문서 — None을 반환해 호출부에서 스킵하게 한다.

    2025-03-04·2025-08-27 필링부터 Vanguard가 같은 날 같은 CIK 아래 accession을
    2건으로 나눠 제출하기 시작했는데, 그 중 하나는 500 펀드가 아예 빠진 형제
    펀드 묶음이다(2026-07-16 실제 필링으로 확인 — 0001104659-25-020311은 504개가
    아니라 1372개 보유종목이 나왔고 내용도 Reliance/Eastman Chemical 등 전혀
    다른 펀드였음, 진짜 500 펀드는 같은 날 accession 0001104659-25-020270).
    이전 코드는 그 날짜의 첫 accession을 무조건 500펀드로 가정해서 실제로는
    이 형제 펀드의 보유종목을 S&P500 스냅샷으로 잘못 저장하고 있었다.
    """
    text, is_legacy = _fetch_filing_document(filing)
    if not re.search(r"500 Index Fund", text, re.IGNORECASE):
        return None
    return _parse_legacy_text(text) if is_legacy else _parse_modern_html(text)


# ── 4. 회사명 → 티커 매칭 ─────────────────────────────────────────────────────
#
# 초기 구현을 2013~2014년 실제 필링으로 테스트해보니 매칭률이 45%밖에 안
# 나와서 원인을 파본 결과, 회사가 상장폐지된 경우(진짜 데이터 부재)보다
# 정규화 코드 자체의 버그가 더 큰 원인이었다(2026-07-16 발견):
#   1) SEC 타이틀의 "/DE", "/MN" 같은 주(州) 등록지 접미사를 안 벗겨냄
#   2) 구두점을 공백이 아니라 완전히 삭제해서 "Amazon.com" → "AMAZONCOM"처럼
#      붙어버려 SEC의 "AMAZON COM INC"와 안 맞음
#   3) 가장 심각한 버그: 같은 회사가 보통주 + 우선주 여러 시리즈로 중복 등록돼
#      있는데(예: WFC, WFC-PA, WFC-PY, WFC-PZ...) dict 컴프리헨션이 마지막에
#      본 걸로 덮어써서 어쩌다 우선주 티커가 최종값으로 남는 경우가 있었다
#      (실제로 Allstate가 "ALL-PB"로 매칭된 사례 발견 — 매칭됐다고 다 맞는
#      게 아니라 조용히 틀린 티커가 들어갈 수 있는 위험한 버그였음).

_SUFFIX_EXPAND = {
    "COMPANY": "CO", "CORPORATION": "CORP", "INCORPORATED": "INC", "LIMITED": "LTD",
}


def _normalize_name(name: str) -> str:
    name = name.upper()
    name = re.sub(r"/[A-Z]{2,3}/?\s*$", "", name)  # 끝의 /DE, /MN 등 주(州) 접미사 제거
    name = name.replace("&", " AND ")               # "&" ↔ "AND" 표기 차이 통일(예: BECTON DICKINSON & CO)
    name = re.sub(r"[^\sA-Z0-9]", " ", name)        # 나머지 구두점은 삭제 대신 공백으로 치환
    name = re.sub(r"\s+", " ", name).strip()
    words = [_SUFFIX_EXPAND.get(w, w) for w in name.split(" ")]
    return " ".join(words)


def _strip_class_suffix(name: str) -> str:
    """'... CLASS A' 같은 종류주 표기 제거 — 폴백 매칭용."""
    return re.sub(r"\s+CLASS\s+[A-Z]$", "", name).strip()


def _is_common_stock_ticker(ticker: str) -> bool:
    """우선주(-PA 등)·워런트 등과 구분하기 위한 "평범한 보통주 티커" 판정."""
    return bool(re.fullmatch(r"[A-Z]{1,5}", ticker))


def _load_ticker_map(force_refresh: bool = False) -> dict[str, str]:
    """
    SEC company_tickers.json(정식 회사명 ↔ 티커) 로드. 30일 캐시.

    같은 정규화 이름에 여러 티커(보통주+우선주 시리즈)가 걸리는 경우,
    "평범한 보통주 티커"(하이픈 없는 순수 알파벳)를 우선한다 — 그렇지 않으면
    JSON 안에서 어느 게 나중에 나오느냐에 따라 우선주 티커로 덮어써질 수 있다.
    """
    if not force_refresh and TICKER_MAP_PATH.exists():
        age_days = (time.time() - TICKER_MAP_PATH.stat().st_mtime) / 86400
        if age_days < _TICKER_MAP_TTL_DAYS:
            raw = json.loads(TICKER_MAP_PATH.read_text(encoding="utf-8"))
        else:
            raw = None
    else:
        raw = None

    if raw is None:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json", headers=_headers(), timeout=30
        )
        resp.raise_for_status()
        raw = resp.json()
        TICKER_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        TICKER_MAP_PATH.write_text(json.dumps(raw), encoding="utf-8")

    ticker_map: dict[str, str] = {}
    for v in raw.values():
        norm = _normalize_name(v["title"])
        ticker = v["ticker"]
        existing = ticker_map.get(norm)
        if existing is None or (_is_common_stock_ticker(ticker) and not _is_common_stock_ticker(existing)):
            ticker_map[norm] = ticker
    return ticker_map


def _load_name_overrides() -> dict[str, str]:
    """수동 유지 이름→티커 오버라이드 (data/universe_name_overrides.csv: raw_name,ticker)."""
    if not NAME_OVERRIDE_CSV.exists():
        return {}
    overrides: dict[str, str] = {}
    with NAME_OVERRIDE_CSV.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            overrides[_normalize_name(row["raw_name"])] = row["ticker"].strip()
    return overrides


def _match_tickers(
    names: list[str], ticker_map: dict[str, str], overrides: dict[str, str]
) -> tuple[dict[str, str], list[str]]:
    """
    회사명 목록 → {raw_name: ticker} 매칭 결과와 미매칭 이름 목록 반환.
    fuzzy 매칭은 참고용 제안만 만들고(_write_unmatched), 자동으로 받아들이지 않는다
    — 잘못된 회사에 매칭되면 팩터·라벨이 조용히 오염되는 게 훨씬 위험하기 때문.
    """
    matched: dict[str, str] = {}
    unmatched: list[str] = []
    for name in names:
        norm = _normalize_name(name)
        norm_noclass = _strip_class_suffix(norm)
        ticker = (
            overrides.get(norm)
            or ticker_map.get(norm)
            or overrides.get(norm_noclass)
            or ticker_map.get(norm_noclass)
        )
        if ticker:
            matched[name] = ticker
        else:
            unmatched.append(name)
    return matched, unmatched


def _write_unmatched_report(all_unmatched: dict[str, list[str]], ticker_map: dict[str, str]) -> None:
    """미매칭 이름 + fuzzy 제안을 검토용 CSV로 저장 (자동 반영 안 됨)."""
    import difflib

    candidates = list(ticker_map.keys())
    rows = []
    for name, snapshots in all_unmatched.items():
        suggestion_norm = difflib.get_close_matches(_normalize_name(name), candidates, n=1, cutoff=_FUZZY_CUTOFF)
        suggested_ticker = ticker_map.get(suggestion_norm[0]) if suggestion_norm else ""
        rows.append({
            "raw_name": name,
            "suggested_ticker": suggested_ticker,
            "seen_in_snapshots": ";".join(sorted(snapshots)),
        })
    df = pd.DataFrame(rows).sort_values("raw_name")
    UNMATCHED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(UNMATCHED_CSV, index=False, encoding="utf-8-sig")
    logger.warning(
        "미매칭 회사명 %d건 → %s (suggested_ticker 확인 후 %s에 raw_name,ticker로 옮겨 담으세요)",
        len(rows), UNMATCHED_CSV.name, NAME_OVERRIDE_CSV.name,
    )


# ── 5. 공개 인터페이스 ────────────────────────────────────────────────────────

def build_universe(start_year: int = 2000, end_year: int | None = None) -> None:
    """
    Vanguard 500 Index Fund 반기보고서 기반 S&P500 과거 시점 스냅샷 생성.

    출력 파일명은 회계기준일(YYYYMM, 06 또는 12)이며 기존 계약(ticker 컬럼)을 유지.
    """
    if end_year is None:
        end_year = pd.Timestamp.today().year

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ticker_map = _load_ticker_map()
    overrides  = _load_name_overrides()

    filings = _get_all_filings()
    filings = [f for f in filings if start_year <= pd.Timestamp(f["date"]).year <= end_year + 1]
    logger.info("Vanguard 500 필링 %d건 대상 (start_year=%d)", len(filings), start_year)

    # 같은 기준일(YYYYMM)에 500 펀드가 아닌 형제 펀드 accession이 섞여 들어올 수
    # 있어(_fund_holdings_or_none 참고) 먼저 기준일별로 묶고, 그 중 실제로 500
    # 펀드 텍스트를 포함한 accession만 채택한다.
    by_key: dict[str, list[tuple[dict, pd.Timestamp]]] = {}
    for filing in filings:
        filed_date = pd.Timestamp(filing["date"])
        # 회계기준일 근사: 3월 제출→전년 12월말, 8월 제출→당해 6월말
        period_end = (
            pd.Timestamp(year=filed_date.year - 1, month=12, day=31)
            if filed_date.month <= 6
            else pd.Timestamp(year=filed_date.year, month=6, day=30)
        )
        key = period_end.strftime("%Y%m")
        by_key.setdefault(key, []).append((filing, period_end))

    all_unmatched: dict[str, list[str]] = {}
    total = 0
    keys = sorted(by_key)

    for i, key in enumerate(keys, 1):
        out_path = SNAPSHOT_DIR / f"{key}.parquet"
        if out_path.exists():
            logger.debug("스킵 (이미 존재): %s", key)
            continue

        candidates = by_key[key]
        period_end = candidates[0][1]
        logger.info(
            "[%d/%d] %s (기준일 %s) 처리 중... (필링 후보 %d건)",
            i, len(keys), key, period_end.date(), len(candidates),
        )

        names: list[str] | None = None
        used_accession: str | None = None
        for filing, _ in candidates:
            try:
                cur_names = _fund_holdings_or_none(filing)
            except Exception as e:
                logger.error("%s 파싱 실패: %s", filing["accession"], e)
                continue
            if cur_names is None:
                logger.debug("%s: 500 Index Fund 아님 — 다음 후보 시도", filing["accession"])
                continue
            names = cur_names
            used_accession = filing["accession"]
            break

        if names is None:
            logger.warning("%s: 500 Index Fund가 포함된 필링을 후보 %d건 중에서 못 찾음 — 스킵", key, len(candidates))
            continue

        matched, unmatched = _match_tickers(names, ticker_map, overrides)
        for name in unmatched:
            all_unmatched.setdefault(name, []).append(key)

        if not matched:
            logger.warning("%s: 매칭된 티커 없음 — 스킵", key)
            continue

        df_out = pd.DataFrame({"ticker": sorted(set(matched.values())), "ref_date": period_end.date()})
        df_out.to_parquet(out_path, index=False)
        logger.info(
            "%s: %d종목 저장 (accession %s, 원본 %d개 중 매칭 %d, 미매칭 %d)",
            key, len(df_out), used_accession, len(names), len(matched), len(unmatched),
        )
        total += 1

    if all_unmatched:
        _write_unmatched_report(all_unmatched, ticker_map)

    logger.info("유니버스 스냅샷 수집 완료: 총 %d개 기준일", total)


def get_universe_by_date(ref_date: str | pd.Timestamp) -> list[str]:
    """ref_date 기준 가장 최근 S&P500 스냅샷의 구성 종목 반환 (생존편향 방지)."""
    if not SNAPSHOT_DIR.exists():
        raise RuntimeError(
            "유니버스 스냅샷 디렉토리 없음. python -m data.build_universe 를 먼저 실행하세요."
        )

    ref = pd.Timestamp(ref_date)
    key_limit = ref.strftime("%Y%m")
    all_files = sorted(SNAPSHOT_DIR.glob("*.parquet"))
    valid = [p for p in all_files if p.stem <= key_limit]
    if not valid:
        raise RuntimeError(
            f"ref_date({ref.date()}) 이전 S&P500 스냅샷 없음. "
            "build_universe()를 더 이른 날짜부터 실행하세요."
        )
    path = valid[-1]
    df = pd.read_parquet(path)
    tickers = df["ticker"].tolist()
    logger.debug("유니버스 로드: %s (%d 종목)", path.stem, len(tickers))
    return tickers


def get_all_tickers_until(end_year: int) -> list[str]:
    """
    start ~ end_year 기간에 S&P500에 한 번이라도 편입된 종목 합집합.
    CNN/LSTM 학습 데이터 수집 시 생존편향 방지용.
    """
    if not SNAPSHOT_DIR.exists():
        return []
    limit = f"{end_year}12"
    all_tickers: set[str] = set()
    for p in SNAPSHOT_DIR.glob("*.parquet"):
        if p.stem <= limit:
            try:
                df = pd.read_parquet(p, columns=["ticker"])
                all_tickers.update(df["ticker"].tolist())
            except Exception:
                pass
    return sorted(all_tickers)


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    end   = int(sys.argv[2]) if len(sys.argv) > 2 else None
    build_universe(start_year=start, end_year=end)
