from __future__ import annotations

import numpy as np


class HealthRulesEngine:
    def __init__(self, thresholds: dict[str, float]) -> None:
        self.healthy_gt = float(thresholds.get("healthy_gt", 90))
        self.subhealthy_gt = float(thresholds.get("subhealthy_gt", 60))
        self.abnormal_gt = float(thresholds.get("abnormal_gt", 30))

    def classify(self, ahi: np.ndarray) -> np.ndarray:
        return np.where(
            ahi > self.healthy_gt,
            "Healthy",
            np.where(
                ahi > self.subhealthy_gt,
                "Sub-Healthy",
                np.where(ahi > self.abnormal_gt, "Abnormal", "Fault"),
            ),
        )
