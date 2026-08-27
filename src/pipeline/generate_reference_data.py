"""One-off generator to restore/extend the dim_asset and new_measurements
reference CSVs to cover all 10 assets used by operational_measurements.csv.

Run manually:

    python -m src.pipeline.generate_reference_data
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DIM_ASSET_PATH = Path("data/raw/dim_asset.csv")
NEW_MEASUREMENTS_PATH = Path("data/raw/new_measurements.csv")

DIM_ASSET_ROWS = [
    {
        "asset_id": "GEN001",
        "asset_name": "Generator 1",
        "asset_type": "Generator",
        "location": "Plant A",
        "manufacturer": "GE",
        "criticality": "High",
        "status": "Active",
        "commissioning_date": "2018-01-15",
    },
    {
        "asset_id": "GEN002",
        "asset_name": "Generator 2",
        "asset_type": "Generator",
        "location": "Plant A",
        "manufacturer": "Mitsubishi",
        "criticality": "High",
        "status": "Active",
        "commissioning_date": "2019-03-20",
    },
    {
        "asset_id": "GEN003",
        "asset_name": "Generator 3",
        "asset_type": "Generator",
        "location": "Plant B",
        "manufacturer": "Siemens",
        "criticality": "Medium",
        "status": "Active",
        "commissioning_date": "2020-05-10",
    },
    {
        "asset_id": "GEN004",
        "asset_name": "Generator 4",
        "asset_type": "Generator",
        "location": "Plant B",
        "manufacturer": "GE",
        "criticality": "Medium",
        "status": "Active",
        "commissioning_date": "2021-07-18",
    },
    {
        "asset_id": "GEN005",
        "asset_name": "Generator 5",
        "asset_type": "Generator",
        "location": "Plant C",
        "manufacturer": "Siemens",
        "criticality": "Low",
        "status": "Active",
        "commissioning_date": "2022-09-01",
    },
    {
        "asset_id": "GEN006",
        "asset_name": "Generator 6",
        "asset_type": "Generator",
        "location": "Plant C",
        "manufacturer": "GE",
        "criticality": "Medium",
        "status": "Active",
        "commissioning_date": "2019-11-05",
    },
    {
        "asset_id": "GEN007",
        "asset_name": "Generator 7",
        "asset_type": "Generator",
        "location": "Plant A",
        "manufacturer": "Mitsubishi",
        "criticality": "High",
        "status": "Active",
        "commissioning_date": "2017-06-22",
    },
    {
        "asset_id": "GEN008",
        "asset_name": "Generator 8",
        "asset_type": "Generator",
        "location": "Plant B",
        "manufacturer": "Siemens",
        "criticality": "Medium",
        "status": "Active",
        "commissioning_date": "2020-12-30",
    },
    {
        "asset_id": "GEN009",
        "asset_name": "Generator 9",
        "asset_type": "Generator",
        "location": "Plant C",
        "manufacturer": "GE",
        "criticality": "Low",
        "status": "Active",
        "commissioning_date": "2023-02-14",
    },
    {
        "asset_id": "GEN010",
        "asset_name": "Generator 10",
        "asset_type": "Generator",
        "location": "Plant A",
        "manufacturer": "Mitsubishi",
        "criticality": "Medium",
        "status": "Active",
        "commissioning_date": "2021-04-09",
    },
]

# One new (unlabeled) reading per asset, spanning healthy -> fault conditions.
NEW_MEASUREMENT_ROWS = [
    {
        "timestamp": "2026-04-01 08:00:00",
        "asset_id": "GEN001",
        "speed": 2995,
        "frequency": 49.8,
        "voltage": 10.8,
        "current": 280,
        "active_power": 4.6,
        "reactive_power": 0.9,
        "power_factor": 0.95,
    },
    {
        "timestamp": "2026-04-01 08:00:00",
        "asset_id": "GEN002",
        "speed": 2988,
        "frequency": 49.6,
        "voltage": 10.5,
        "current": 340,
        "active_power": 4.0,
        "reactive_power": 1.2,
        "power_factor": 0.90,
    },
    {
        "timestamp": "2026-04-01 08:00:00",
        "asset_id": "GEN003",
        "speed": 2972,
        "frequency": 49.0,
        "voltage": 10.0,
        "current": 430,
        "active_power": 3.4,
        "reactive_power": 1.8,
        "power_factor": 0.77,
    },
    {
        "timestamp": "2026-04-01 08:00:00",
        "asset_id": "GEN004",
        "speed": 3000,
        "frequency": 50.0,
        "voltage": 11.0,
        "current": 250,
        "active_power": 4.9,
        "reactive_power": 0.7,
        "power_factor": 0.98,
    },
    {
        "timestamp": "2026-04-01 08:00:00",
        "asset_id": "GEN005",
        "speed": 2992,
        "frequency": 49.7,
        "voltage": 10.7,
        "current": 310,
        "active_power": 4.2,
        "reactive_power": 1.0,
        "power_factor": 0.92,
    },
    {
        "timestamp": "2026-04-01 08:00:00",
        "asset_id": "GEN006",
        "speed": 3001,
        "frequency": 50.0,
        "voltage": 11.0,
        "current": 260,
        "active_power": 4.8,
        "reactive_power": 0.6,
        "power_factor": 0.99,
    },
    {
        "timestamp": "2026-04-01 08:00:00",
        "asset_id": "GEN007",
        "speed": 2960,
        "frequency": 48.6,
        "voltage": 9.9,
        "current": 460,
        "active_power": 3.1,
        "reactive_power": 2.0,
        "power_factor": 0.73,
    },
    {
        "timestamp": "2026-04-01 08:00:00",
        "asset_id": "GEN008",
        "speed": 2980,
        "frequency": 49.4,
        "voltage": 10.4,
        "current": 360,
        "active_power": 3.9,
        "reactive_power": 1.4,
        "power_factor": 0.88,
    },
    {
        "timestamp": "2026-04-01 08:00:00",
        "asset_id": "GEN009",
        "speed": 2998,
        "frequency": 49.9,
        "voltage": 10.9,
        "current": 255,
        "active_power": 4.7,
        "reactive_power": 0.8,
        "power_factor": 0.97,
    },
    {
        "timestamp": "2026-04-01 08:00:00",
        "asset_id": "GEN010",
        "speed": 2950,
        "frequency": 48.3,
        "voltage": 9.6,
        "current": 490,
        "active_power": 2.8,
        "reactive_power": 2.2,
        "power_factor": 0.66,
    },
]


def main() -> None:
    dim_asset_df = pd.DataFrame.from_records(DIM_ASSET_ROWS)
    DIM_ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dim_asset_df.to_csv(DIM_ASSET_PATH, index=False)
    print(f"Wrote {len(dim_asset_df)} records to {DIM_ASSET_PATH}")

    new_measurements_df = pd.DataFrame.from_records(NEW_MEASUREMENT_ROWS)
    NEW_MEASUREMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_measurements_df.to_csv(NEW_MEASUREMENTS_PATH, index=False)
    print(f"Wrote {len(new_measurements_df)} records to {NEW_MEASUREMENTS_PATH}")


if __name__ == "__main__":
    main()
