"""Unit tests for the Health Index calculator and classification engine."""

from __future__ import annotations

import math

import pytest

from src.ml.health_index import HealthIndexCalculator
from src.rules.health_classification import (
    ClassificationThresholds,
    HealthClassificationEngine,
)


def test_health_index_at_zero_distance_is_one() -> None:
    calculator = HealthIndexCalculator(b=0.5)
    hi = calculator.calculate_health_index(0.0)
    assert hi == pytest.approx(1.0)


def test_ahi_score_at_zero_distance_is_100() -> None:
    calculator = HealthIndexCalculator(b=0.5)
    ahi = calculator.calculate_ahi_score(0.0)
    assert ahi == pytest.approx(100.0)


def test_health_index_matches_formula() -> None:
    b = 0.5
    md = 2.0
    calculator = HealthIndexCalculator(b=b)
    expected = 1.0 - (2.0 / math.pi) * math.atan(b * md)
    assert calculator.calculate_health_index(md) == pytest.approx(expected)


def test_health_index_decreases_as_distance_increases() -> None:
    calculator = HealthIndexCalculator(b=0.5)
    hi_small = calculator.calculate_health_index(1.0)
    hi_large = calculator.calculate_health_index(10.0)
    assert hi_large < hi_small


def test_negative_distance_raises() -> None:
    calculator = HealthIndexCalculator(b=0.5)
    with pytest.raises(ValueError):
        calculator.calculate_health_index(-1.0)


def test_calculate_batch_matches_single_values() -> None:
    calculator = HealthIndexCalculator(b=0.5)
    distances = [0.0, 1.0, 5.0]
    hi_batch, ahi_batch = calculator.calculate_batch(distances)
    for i, md in enumerate(distances):
        assert hi_batch[i] == pytest.approx(calculator.calculate_health_index(md))
        assert ahi_batch[i] == pytest.approx(calculator.calculate_ahi_score(md))


# ---------------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------------


def test_classification_thresholds_validate_ordering() -> None:
    with pytest.raises(ValueError):
        ClassificationThresholds(healthy=50, subhealthy=60, abnormal=30)


@pytest.mark.parametrize(
    "ahi_score,expected_status",
    [
        (95, "Healthy"),
        (90.01, "Healthy"),
        (90, "Sub-Healthy"),
        (75, "Sub-Healthy"),
        (60, "Abnormal"),
        (45, "Abnormal"),
        (30, "Fault"),
        (10, "Fault"),
    ],
)
def test_classify_default_thresholds(ahi_score: float, expected_status: str) -> None:
    engine = HealthClassificationEngine()
    assert engine.classify(ahi_score) == expected_status


def test_classify_batch_returns_expected_labels() -> None:
    engine = HealthClassificationEngine()
    results = engine.classify_batch([95, 75, 45, 10])
    assert results == ["Healthy", "Sub-Healthy", "Abnormal", "Fault"]
