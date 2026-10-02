"""
시계열 면적 분석: RVI 산출, 치유 패턴 분류, 안정화일 추정.

앱과 RVI 분석 노트북이 반드시 같은 규칙을 쓰도록 이 모듈 하나에만 구현한다.
torch에 의존하지 않으므로 단독으로 테스트할 수 있다.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit

from wound.config import (
    PLATEAU_THR,
    REBOUND_THR,
    REF_DAYS,
    STABLE_HORIZON_DAYS,
    STABLE_RATIO,
)

# 안정화일 추정 방식
STABLE_OBSERVED     = "observed"      # 측정값이 이미 기준 이하
STABLE_EXTRAPOLATED = "extrapolated"  # 추세로 외삽한 예상일
STABLE_NOT_REACHED  = "not_reached"   # 현재 추세로는 탐색 범위 내 도달 불가
STABLE_UNKNOWN      = "unknown"       # 데이터 부족


def compute_rvi(areas: Sequence[float], days: Sequence[int],
                ref_days: int = REF_DAYS) -> float:
    """
    RVI(Recovery Velocity Index, 0~100) 산출.

    RVI = clip((A0 - A_last) / A0 × ref_days / (T_last - T0) × 100, 0, 100)

    Args:
        areas: 각 시점의 상처 면적 (px)
        days: 각 시점의 측정일
        ref_days: 정규화 기준일 (기본 14일 → RVI 100 = 2주 내 완전 회복 속도)
    Returns:
        float: RVI 점수. 데이터가 2개 미만이거나 A0 <= 0이면 0.
    """
    if len(areas) < 2 or areas[0] <= 0:
        return 0.0
    t_elapsed = days[-1] - days[0]
    if t_elapsed <= 0:
        return 0.0
    total_reduction = (areas[0] - areas[-1]) / areas[0]   # 음수 = 악화
    rvi = total_reduction / t_elapsed * ref_days * 100
    return float(np.clip(rvi, 0, 100))


def classify_pattern(areas: Sequence[float],
                     rebound_thr: float = REBOUND_THR,
                     plateau_thr: float = PLATEAU_THR) -> str:
    """
    시계열 면적으로 치유 패턴 분류.

    Rules (구간 i = areas[i] → areas[i+1]):
      Rebound : 중간 구간(첫 구간과 마지막 구간 제외)에서 면적이 rebound_thr 초과 증가.
                7개 시점(Day 0~18)이면 Day 3→6 ~ Day 12→15 구간.
                시점이 3개뿐이라 중간 구간이 없으면 모든 구간을 검사한다.
      Plateau : 후반 최대 3구간의 변화율이 모두 plateau_thr 미만
      Normal  : 위 두 조건 미해당
    면적이 0인 시점에서 시작하는 구간은 변화율을 정의할 수 없어 건너뛴다.

    Args:
        areas: 시계열 면적 (측정일 오름차순)
        rebound_thr: rebound 판정 임계값
        plateau_thr: plateau 판정 임계값
    Returns:
        str: 'normal' | 'plateau' | 'rebound' | 'unknown'(시점 3개 미만)
    """
    n = len(areas)
    if n < 3:
        return "unknown"

    rebound_intervals = range(1, n - 2) if n >= 4 else range(0, n - 1)
    for i in rebound_intervals:
        if areas[i] > 0 and (areas[i + 1] - areas[i]) / areas[i] > rebound_thr:
            return "rebound"

    late_changes = [
        abs(areas[i + 1] - areas[i]) / areas[i]
        for i in range(max(0, n - 4), n - 1)
        if areas[i] > 0
    ]
    if late_changes and all(c < plateau_thr for c in late_changes):
        return "plateau"

    return "normal"


def _exp_decay(t, a, b, c):
    return a * np.exp(-b * t) + c


def estimate_stable_day(areas: Sequence[float], days: Sequence[int],
                        stable_ratio: float = STABLE_RATIO,
                        horizon: int = STABLE_HORIZON_DAYS
                        ) -> Tuple[Optional[int], str]:
    """
    안정화일 추정. '안정화' = 상처 면적이 초기 면적의 stable_ratio(기본 5%) 이하.

    1. 측정값 중 기준 이하인 첫 시점이 있으면 그 날짜 (observed)
    2. 없으면 지수 감소 a·exp(-b·t)+c 피팅으로 외삽 (시점 3개 이상)
       피팅이 불가능하면 마지막 두 시점의 선형 추세로 외삽 (extrapolated)
    3. 마지막 측정일 + horizon 안에 도달하지 않으면 (None, 'not_reached')

    Args:
        areas: 시계열 면적 (측정일 오름차순)
        days: 측정일
        stable_ratio: 초기 면적 대비 안정화 기준 비율
        horizon: 외삽 최대 탐색 일수
    Returns:
        (day, method): method는 'observed' | 'extrapolated' | 'not_reached' | 'unknown'.
        day는 observed/extrapolated일 때만 int, 그 외에는 None.
    """
    if len(areas) < 2 or areas[0] <= 0:
        return None, STABLE_UNKNOWN

    a0 = float(areas[0])
    target = a0 * stable_ratio
    days = [int(d) for d in days]
    last_day = days[-1]

    for a, d in zip(areas[1:], days[1:]):
        if a <= target:
            return d, STABLE_OBSERVED

    if len(areas) >= 3:
        try:
            popt, _ = curve_fit(
                _exp_decay, np.asarray(days, dtype=float), np.asarray(areas, dtype=float),
                p0=[a0, 0.05, a0 * 0.05], maxfev=5000,
                bounds=([0, 1e-6, 0], [a0 * 10, 2, a0]),
            )
            for d in range(last_day + 1, last_day + horizon + 1):
                if _exp_decay(d, *popt) <= target:
                    return d, STABLE_EXTRAPOLATED
            return None, STABLE_NOT_REACHED
        except (RuntimeError, ValueError):
            pass  # 피팅 실패 → 선형 외삽

    rate = (areas[-1] - areas[-2]) / max(1, days[-1] - days[-2])
    if rate >= 0:
        return None, STABLE_NOT_REACHED
    days_needed = int(np.ceil((areas[-1] - target) / -rate))
    if days_needed > horizon:
        return None, STABLE_NOT_REACHED
    return last_day + days_needed, STABLE_EXTRAPOLATED
