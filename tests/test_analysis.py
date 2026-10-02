import pytest

from wound.analysis import (
    STABLE_EXTRAPOLATED,
    STABLE_NOT_REACHED,
    STABLE_OBSERVED,
    STABLE_UNKNOWN,
    classify_pattern,
    compute_rvi,
    estimate_stable_day,
)

DAYS = [0, 3, 6, 9, 12, 15, 18]
# Data.ipynb의 의사 시계열 템플릿
RATIOS_NORMAL  = [1.00, 0.75, 0.55, 0.40, 0.28, 0.18, 0.10]
RATIOS_PLATEAU = [1.00, 0.78, 0.60, 0.45, 0.44, 0.44, 0.40]
RATIOS_REBOUND = [1.00, 0.78, 0.60, 0.48, 0.52, 0.35, 0.22]


def scaled(ratios, a0=10000):
    return [r * a0 for r in ratios]


# ── compute_rvi ─────────────────────────────
def test_rvi_full_recovery_in_ref_days_is_100():
    assert compute_rvi([1000, 0], [0, 14]) == pytest.approx(100.0)


def test_rvi_half_recovery_in_ref_days_is_50():
    assert compute_rvi([1000, 500], [0, 14]) == pytest.approx(50.0)


def test_rvi_worsening_is_clipped_to_zero():
    assert compute_rvi([1000, 1500], [0, 14]) == 0.0


@pytest.mark.parametrize("areas, days", [([1000], [0]), ([0, 0], [0, 3]), ([1000, 500], [3, 3])])
def test_rvi_degenerate_inputs_return_zero(areas, days):
    assert compute_rvi(areas, days) == 0.0


# ── classify_pattern ────────────────────────
@pytest.mark.parametrize("ratios, expected", [
    (RATIOS_NORMAL, "normal"),
    (RATIOS_PLATEAU, "plateau"),
    (RATIOS_REBOUND, "rebound"),
])
def test_pattern_templates(ratios, expected):
    assert classify_pattern(scaled(ratios)) == expected


def test_pattern_too_few_points_is_unknown():
    assert classify_pattern([100, 90]) == "unknown"


def test_pattern_ignores_increase_in_first_interval():
    # 첫 구간(Day 0→3)의 일시적 증가는 rebound 판정에서 제외 (노트북 규칙과 동일)
    areas = scaled([1.00, 1.10, 0.80, 0.60, 0.40, 0.25, 0.10])
    assert classify_pattern(areas) == "normal"


def test_pattern_ignores_increase_in_last_interval():
    areas = scaled([1.00, 0.75, 0.55, 0.40, 0.28, 0.18, 0.25])
    assert classify_pattern(areas) == "normal"


def test_pattern_three_points_checks_all_intervals_for_rebound():
    assert classify_pattern([1000, 1100, 900]) == "rebound"


def test_pattern_skips_interval_starting_from_zero_area():
    # 0에서 시작하는 구간은 변화율을 정의할 수 없어 무시
    areas = [1000, 700, 400, 0, 50, 0, 0]
    assert classify_pattern(areas) == "normal"


# ── estimate_stable_day ─────────────────────
def test_stable_day_observed_when_area_reaches_threshold():
    areas = [1000, 600, 300, 40, 20]
    assert estimate_stable_day(areas, [0, 3, 6, 9, 12]) == (9, STABLE_OBSERVED)


def test_stable_day_not_observed_from_a_single_flat_interval():
    # 이전 버그: 변화율 5% 미만 구간이 하나라도 있으면 면적이 80%여도 '관측 안정화'로 판정
    areas = [1000, 800, 790, 500, 300]
    day, method = estimate_stable_day(areas, [0, 3, 6, 9, 12])
    assert method != STABLE_OBSERVED


def test_stable_day_extrapolated_for_decaying_series():
    areas = scaled(RATIOS_NORMAL)
    day, method = estimate_stable_day(areas, DAYS)
    assert method == STABLE_EXTRAPOLATED
    assert day > DAYS[-1]


def test_stable_day_linear_extrapolation_with_two_points():
    # 0→10일 동안 1000→500 (−50/day), 목표 50까지 9일 더 → Day 19
    assert estimate_stable_day([1000, 500], [0, 10]) == (19, STABLE_EXTRAPOLATED)


def test_stable_day_not_reached_when_not_decreasing():
    assert estimate_stable_day([1000, 1000], [0, 10]) == (None, STABLE_NOT_REACHED)


def test_stable_day_not_reached_for_plateau():
    areas = [1000, 600, 450, 440, 440, 440, 440]
    assert estimate_stable_day(areas, DAYS) == (None, STABLE_NOT_REACHED)


def test_stable_day_unknown_with_single_point():
    assert estimate_stable_day([1000], [0]) == (None, STABLE_UNKNOWN)
