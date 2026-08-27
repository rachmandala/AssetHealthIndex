import numpy as np

from asset_health_index.health_index import calculate_ahi
from asset_health_index.rules import HealthRulesEngine


def test_ahi_range_and_monotonicity():
    md = np.array([0.0, 1.0, 2.0, 5.0])
    ahi = calculate_ahi(md, b=0.1)
    assert np.all((ahi >= 0) & (ahi <= 100))
    assert np.all(np.diff(ahi) <= 0)


def test_rules_thresholds():
    rules = HealthRulesEngine({"healthy_gt": 90, "subhealthy_gt": 60, "abnormal_gt": 30})
    ahi = np.array([95.0, 85.0, 45.0, 10.0])
    labels = rules.classify(ahi)
    assert labels.tolist() == ["Healthy", "Sub-Healthy", "Abnormal", "Fault"]
