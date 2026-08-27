import pandas as pd

from asset_health_index.pipeline import AHIPipeline


def test_pipeline_generates_expected_outputs(tmp_path):
    data = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 00:00:00",
                "2024-01-01 01:00:00",
                "2024-01-01 02:00:00",
                "2024-01-01 03:00:00",
                "2024-01-01 04:00:00",
                "2024-01-01 05:00:00",
                "2024-01-01 06:00:00",
                "2024-01-01 07:00:00",
            ],
            "asset_id": ["GEN-001", "GEN-001", "GEN-002", "GEN-002", "GEN-001", "GEN-001", "GEN-002", "GEN-002"],
            "speed": [3000, 3010, 2990, 2980, 2800, 2790, 2500, 2480],
            "frequency": [50, 50.1, 50, 49.8, 49.7, 49.6, 48, 47.9],
            "voltage": [11000, 11020, 10950, 10900, 10800, 10750, 10000, 9950],
            "current": [100, 101, 102, 105, 120, 122, 140, 142],
            "active_power": [5000, 5050, 4900, 4800, 4500, 4450, 3500, 3400],
            "reactive_power": [1000, 980, 1100, 1150, 1300, 1350, 1800, 1850],
            "power_factor": [0.95, 0.955, 0.94, 0.93, 0.90, 0.89, 0.80, 0.79],
            "label": ["Healthy", "Healthy", "Sub-Healthy", "Sub-Healthy", "Abnormal", "Abnormal", "Fault", "Fault"],
        }
    )
    input_csv = tmp_path / "input.csv"
    out_dir = tmp_path / "out"
    data.to_csv(input_csv, index=False)

    pipeline = AHIPipeline(config_path="config/default.yaml")
    pipeline.run(input_csv=input_csv, output_dir=out_dir)

    assert (out_dir / "predictions.csv").exists()
    assert (out_dir / "predictions.json").exists()
    assert (out_dir / "health_trends.csv").exists()
    assert (out_dir / "powerbi_dataset.csv").exists()
    assert (out_dir / "model_evaluation.json").exists()
    assert (out_dir / "data_quality_report.json").exists()
