"""
미국 주식 호가단위(틱 사이즈) 계산.

SEC Rule 612(서브페니 규정) 기준: $1 미만은 $0.0001, $1 이상은 $0.01 균일
(2026-07-16 확정, config.settings.TICK_SIZE_TABLE 참고). KRX처럼 가격대별
계단식 구간이 여러 개가 아니라 2단계뿐이라 계산 자체는 국내보다 단순하다.

참고: SEC가 유동성 높은 종목 대상 $0.005 틱을 추가하는 개정안을 2026-11
첫 영업일 시행 예정으로 유예해뒀다 — 실거래 전환 시점에 재확인 필요.

가격은 KRW(정수)와 달리 소수점(센트 이하)이 있어, 부동소수점 오차가
쌓이지 않도록 매 연산 결과를 소수 4자리(최소 틱 단위)로 반올림한다.
"""

from __future__ import annotations

import math

from config.settings import TICK_SIZE_TABLE

_ROUND_DECIMALS = 4  # 최소 틱($0.0001)과 동일한 정밀도


def get_tick_size(price: float) -> float:
    """주어진 가격에 해당하는 미국 주식 호가단위 반환."""
    for upper_bound, tick in TICK_SIZE_TABLE:
        if price < upper_bound:
            return tick
    return TICK_SIZE_TABLE[-1][1]


def round_to_tick(price: float) -> float:
    """호가단위에 맞춰 가격을 내림 처리 (호가단위 배수가 아닌 가격은 유효하지 않음)."""
    tick = get_tick_size(price)
    return round(math.floor(price / tick) * tick, _ROUND_DECIMALS)


def price_plus_ticks(price: float, n_ticks: int) -> float:
    """price를 호가단위에 맞춘 뒤 n_ticks만큼 위로 이동한 가격 반환."""
    p = round_to_tick(price)
    for _ in range(n_ticks):
        p = round(p + get_tick_size(p), _ROUND_DECIMALS)
    return p


def price_minus_ticks(price: float, n_ticks: int) -> float:
    """price를 호가단위에 맞춘 뒤 n_ticks만큼 아래로 이동한 가격 반환."""
    p = round_to_tick(price)
    for _ in range(n_ticks):
        # 호가단위는 구간에 따라 달라지므로, 한 틱 내려간 뒤의 구간 기준으로 다시 계산
        p = round(p - get_tick_size(p - 0.0001), _ROUND_DECIMALS)
    return max(p, get_tick_size(0))
