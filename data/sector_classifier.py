"""
섹터 분류기 — GICS 11개 섹터, 위키피디아 S&P500 표의 "GICS Sector" 컬럼 기반.

2026-07-16 결정: 과거 시점에도 "현재" GICS 분류를 근사 적용한다(대부분 종목은
섹터가 잘 안 바뀌므로 MVP로는 무난). 국내 버전의 KRX 업종→커스텀 섹터 재매핑
(custom_sector_mapping.csv 오버라이드 포함)은 필요 없다 — GICS 자체가 이미
깔끔한 11개 섹터라 그대로 쓰면 된다.

한계: 지금 S&P500에 없는(상장폐지·인수합병된) 과거 구성종목은 위키 표에 없어
섹터가 미분류로 남는다. backtest/engine.py의 섹터 상한 로직이 "미분류 종목은
상한 적용 제외하고 포함"으로 이미 관대하게 처리하므로 문제 없음.

컬럼명은 "custom_sector"로 국내 버전과 동일하게 유지 — engine.py,
build_factor_dataset.py, agents/*.py가 전부 이 이름을 하드코딩 참조 중.

출력: data/sp500_gics_sector.csv (columns: ticker, custom_sector)

실행: python -m data.sector_classifier
"""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from config.settings import DATA_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_CSV = DATA_DIR / "sp500_gics_sector.csv"
_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def build_sector_csv(ref_date: str | None = None) -> pd.DataFrame:
    """
    현재 S&P500 종목의 GICS 섹터 조회 후 CSV 저장.

    ref_date는 국내 버전과의 함수 시그니처 호환용 인자 — GICS 근사 적용
    방침(과거에도 현재 분류 그대로 사용)상 실제로는 쓰이지 않는다.
    """
    resp = requests.get(
        _SP500_WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (quant-data-collector)"}, timeout=15
    )
    resp.raise_for_status()
    df = pd.read_html(StringIO(resp.text))[0]
    df = df.rename(columns={"Symbol": "ticker", "GICS Sector": "custom_sector"})[["ticker", "custom_sector"]]
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.info("GICS 섹터 CSV 저장 완료: %s (%d 종목)", OUTPUT_CSV, len(df))
    return df


def load_sector_map(ref_date: str | None = None) -> pd.DataFrame:
    """
    저장된 CSV 로드. 없으면 build_sector_csv() 실행 후 반환.

    Returns
    -------
    DataFrame columns: ticker, custom_sector
    """
    if not OUTPUT_CSV.exists():
        logger.info("섹터 CSV 없음 — 신규 생성")
        return build_sector_csv(ref_date)
    return pd.read_csv(OUTPUT_CSV, dtype=str)


def get_custom_sector(ticker: str, ref_date: str | None = None) -> str | None:
    """단일 종목의 GICS 섹터 반환. 미분류(주로 상장폐지 종목)면 None."""
    df = load_sector_map(ref_date)
    row = df[df["ticker"] == ticker]
    if row.empty:
        return None
    val = row.iloc[0]["custom_sector"]
    return None if pd.isna(val) else str(val)


if __name__ == "__main__":
    result = build_sector_csv()
    print(result.to_string(index=False))
