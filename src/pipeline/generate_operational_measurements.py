"""One-off generator for the synthetic operational_measurements.csv dataset.

Produces a labeled diesel generator dataset with a fixed class distribution
and per-class feature ranges, for BP Neural Network training and healthy
baseline construction. Not part of the runtime pipeline; run manually:

    python -m src.pipeline.generate_operational_measurements
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path("data/raw/operational_measurements.csv")

ASSET_IDS = [f"GEN{str(i).zfill(3)}" for i in range(1, 11)]

CLASS_COUNTS = {
    "Healthy": 400,
    "Sub-Healthy": 600,
    "Abnormal": 300,
    "Fault": 100,
}

FEATURE_RANGES = {
    "Healthy": {
        "speed": (2995, 3005),
        "frequency": (49.9, 50.1),
        "voltage": (10.9, 11.1),
        "current": (240, 280),
        "active_power": (4.6, 5.0),
        "reactive_power": (0.5, 0.8),
        "power_factor": (0.97, 1.00),
    },
    "Sub-Healthy": {
        "speed": (2985, 2995),
        "frequency": (49.7, 49.9),
        "voltage": (10.7, 10.9),
        "current": (280, 320),
        "active_power": (4.2, 4.6),
        "reactive_power": (0.8, 1.1),
        "power_factor": (0.93, 0.97),
    },
    "Abnormal": {
        "speed": (2970, 2985),
        "frequency": (49.2, 49.7),
        "voltage": (10.2, 10.7),
        "current": (320, 400),
        "active_power": (3.7, 4.2),
        "reactive_power": (1.1, 1.6),
        "power_factor": (0.85, 0.93),
    },
    "Fault": {
        "speed": (2940, 2970),
        "frequency": (48.0, 49.2),
        "voltage": (9.0, 10.2),
        "current": (400, 550),
        "active_power": (2.5, 3.7),
        "reactive_power": (1.6, 2.5),
        "power_factor": (0.60, 0.85),
    },
}

PERIOD_DAYS = 90
START_DATE = datetime(2026, 1, 1, 0, 0, 0)
RANDOM_STATE = 42


def generate_dataset(random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate the synthetic labeled operational measurement dataset."""
    rng = np.random.default_rng(random_state)

    labels = np.array(
        [label for label, count in CLASS_COUNTS.items() for _ in range(count)]
    )
    rng.shuffle(labels)
    n_records = len(labels)

    asset_ids = rng.choice(ASSET_IDS, size=n_records)

    period_seconds = PERIOD_DAYS * 24 * 60 * 60
    max_slots = period_seconds // 300
    # Guarantee unique (asset_id, timestamp) pairs by resampling on collision.
    used_pairs: set[tuple[str, int]] = set()
    offsets = np.empty(n_records, dtype=np.int64)
    for i, asset_id in enumerate(asset_ids):
        while True:
            slot = int(rng.integers(0, max_slots))
            key = (asset_id, slot)
            if key not in used_pairs:
                used_pairs.add(key)
                offsets[i] = slot * 300
                break
    timestamps = [START_DATE + timedelta(seconds=int(o)) for o in offsets]

    records = []
    for label, asset_id, timestamp in zip(labels, asset_ids, timestamps):
        ranges = FEATURE_RANGES[label]
        records.append(
            {
                "timestamp": timestamp,
                "asset_id": asset_id,
                "speed": round(rng.uniform(*ranges["speed"])),
                "frequency": round(rng.uniform(*ranges["frequency"]), 1),
                "voltage": round(rng.uniform(*ranges["voltage"]), 1),
                "current": round(rng.uniform(*ranges["current"])),
                "active_power": round(rng.uniform(*ranges["active_power"]), 1),
                "reactive_power": round(rng.uniform(*ranges["reactive_power"]), 1),
                "power_factor": round(rng.uniform(*ranges["power_factor"]), 2),
                "label": label,
            }
        )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def main() -> None:
    df = generate_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} records to {OUTPUT_PATH}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
