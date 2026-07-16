"""
NYSE 영업일 계산 유틸리티.
모든 날짜 계산은 달력일이 아닌 영업일(Trading Days) 기준.

국내 버전(XKRX)에서 캘린더만 NYSE로 교체 — 날짜 단위 연산(어느 날이 영업일인지,
N영업일 뒤가 언제인지)은 서머타임과 무관하다(개장/마감 "시각"만 연 2회 바뀌지,
그날이 영업일이라는 사실 자체는 안 바뀜). 서머타임 때문에 실제로 손봐야 하는 건
main.py의 장중 스케줄링(09:30/16:00 등 시각 트리거)이고, 이건 실운용 전환
단계에서 별도로 다룬다(지금은 백테스팅 범위 밖).

성능 최적화:
  모듈 로드 시 2010-01-01 ~ 2035-12-31 NYSE 영업일 전체를 한 번에 pre-compute.
  이후 모든 calendar 호출은 numpy 바이너리 서치 → O(log n) 즉시 처리.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

_NYSE = mcal.get_calendar("NYSE")

# ── 전체 NYSE 영업일 사전 로드 (모듈 최초 임포트 시 1회) ──────────────────────
def _build_nyse_days() -> np.ndarray:
    sched = _NYSE.schedule(start_date="2010-01-01", end_date="2035-12-31")
    idx   = mcal.date_range(sched, frequency="1D").normalize().tz_localize(None)
    # mcal.date_range가 open/close 양쪽을 반환할 수 있어 normalize 후 중복 발생
    # → np.unique로 제거 (정렬 보장)
    return np.unique(idx.values.astype("datetime64[D]"))

_NYSE_DAYS: np.ndarray = _build_nyse_days()   # shape (N,) dtype datetime64[D]


def _to_day(dt: str | pd.Timestamp) -> np.datetime64:
    return np.datetime64(pd.Timestamp(dt).date(), "D")


def get_trading_days(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DatetimeIndex:
    """start~end 사이 NYSE 영업일 목록 반환 (양 끝 포함)."""
    s = _to_day(start)
    e = _to_day(end)
    lo = int(np.searchsorted(_NYSE_DAYS, s, side="left"))
    hi = int(np.searchsorted(_NYSE_DAYS, e, side="right"))
    return pd.DatetimeIndex(pd.to_datetime(_NYSE_DAYS[lo:hi]))


def add_trading_days(dt: str | pd.Timestamp, n: int) -> pd.Timestamp:
    """dt 기준 n 영업일 후 날짜 반환 (n < 0이면 이전).

    n=0 → dt 반환 (단, dt가 영업일이 아닌 경우에도 그대로 반환).
    n>0 → dt 이후 첫 번째 영업일 기준 n번째 영업일.
    n<0 → dt 이전 영업일 기준 |n|번째 이전.
    """
    d = _to_day(dt)
    pos = int(np.searchsorted(_NYSE_DAYS, d, side="right"))   # dt 이후 첫 번째 위치
    if n == 0:
        return pd.Timestamp(dt)
    if n > 0:
        target = pos + n - 1           # pos는 이미 dt 초과 첫 번째 인덱스
    else:                              # n < 0
        # pos-1은 dt 이전(또는 dt 포함 최후) 인덱스
        before = pos - 1              # dt 이하 마지막 인덱스
        target = before + n           # n이 음수이므로 감소
    if target < 0 or target >= len(_NYSE_DAYS):
        raise ValueError(f"add_trading_days: out of range ({dt!r} + {n})")
    return pd.Timestamp(_NYSE_DAYS[target].astype("datetime64[ns]"))


def prev_trading_day(dt: str | pd.Timestamp) -> pd.Timestamp:
    """dt 직전 영업일 반환."""
    return add_trading_days(dt, -1)


def next_trading_day(dt: str | pd.Timestamp) -> pd.Timestamp:
    """dt 직후 영업일 반환."""
    return add_trading_days(dt, 1)


def count_trading_days(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> int:
    """start 이후~end 포함 사이의 영업일 수 반환 (start 미포함, end 포함)."""
    s = _to_day(start)
    e = _to_day(end)
    lo = int(np.searchsorted(_NYSE_DAYS, s, side="right"))   # start 초과 첫 인덱스
    hi = int(np.searchsorted(_NYSE_DAYS, e, side="right"))   # end 이하 마지막+1
    return max(0, hi - lo)


def is_trading_day(dt: str | pd.Timestamp) -> bool:
    d   = _to_day(dt)
    pos = int(np.searchsorted(_NYSE_DAYS, d, side="left"))
    return pos < len(_NYSE_DAYS) and _NYSE_DAYS[pos] == d


def get_first_monday_of_month(year: int, month: int) -> pd.Timestamp:
    """해당 연월의 첫째 월요일 반환."""
    d = pd.Timestamp(year=year, month=month, day=1)
    days_until_monday = (7 - d.dayofweek) % 7
    return d + pd.Timedelta(days=days_until_monday)


def get_semiannual_dates(year: int) -> list[pd.Timestamp]:
    """연간 반기 교체 기준일 목록 (1월·7월 첫째 월요일)."""
    return [get_first_monday_of_month(year, m) for m in [1, 7]]


def get_friday(dt: str | pd.Timestamp) -> pd.Timestamp:
    """dt가 속한 주의 금요일 반환."""
    dt = pd.Timestamp(dt)
    days_to_friday = (4 - dt.dayofweek) % 7
    friday = dt + pd.Timedelta(days=days_to_friday)
    while not is_trading_day(friday):
        friday -= pd.Timedelta(days=1)
    return friday


def last_friday_before(dt: str | pd.Timestamp) -> pd.Timestamp:
    """dt 직전(또는 당일) 금요일 영업일 반환."""
    dt = pd.Timestamp(dt)
    days_back = (dt.dayofweek - 4) % 7
    friday = dt - pd.Timedelta(days=days_back)
    while not is_trading_day(friday):
        friday -= pd.Timedelta(days=1)
    return friday
