from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .baseline import build_healthy_baseline
from .config import load_config
from .health_index import calculate_ahi
from .mahalanobis import mahalanobis_distance_batch
from .model import BPNeuralNetwork
from .preprocessing import Preprocessor
from .recommendations import recommendation_for_class
from .rules import HealthRulesEngine
from .validation import DataValidator


class AHIPipeline:
    def __init__(self, config_path: str | Path) -> None:
        self.config = load_config(config_path)
        self.features = self.config["features"]
        self.labels_cfg = self.config["labels"]

    def run(self, input_csv: str | Path, output_dir: str | Path) -> dict:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(input_csv)

        validator = DataValidator(
            features=self.features,
            label_col="label",
            ranges=self.config["validation"]["ranges"],
        )
        quality_report = validator.validate(df)
        quality_report.save(out_dir / "data_quality_report.json")

        pre = Preprocessor(
            features=self.features,
            outlier_quantiles=tuple(self.config["preprocessing"]["outlier_quantiles"]),
            normalize=bool(self.config["preprocessing"]["normalize"]),
        )
        x = pre.fit_transform(df)
        pre.save(out_dir / "preprocessor.joblib")

        y = df["label"].astype(str).to_numpy()
        bp = BPNeuralNetwork(self.config["model"], random_state=int(self.config["random_state"]))
        metrics = bp.train(x, y)
        bp.save(out_dir / "bp_model.joblib")

        healthy_label = self.labels_cfg["healthy"]
        healthy_mask = df["label"].astype(str) == healthy_label
        baseline = build_healthy_baseline(
            x_healthy=x[healthy_mask.to_numpy()],
            regularization=float(self.config["health_index"].get("covariance_regularization", 1e-6)),
        )
        baseline.save(out_dir / "healthy_baseline.joblib")

        bp_prediction = bp.predict(x)
        probabilities = bp.predict_proba(x)

        proba_df = pd.DataFrame(probabilities, columns=[f"{c.lower().replace('-', '').replace(' ', '_')}_probability" for c in bp.model.classes_])
        expected_cols = [
            "healthy_probability",
            "subhealthy_probability",
            "abnormal_probability",
            "fault_probability",
        ]
        for col in expected_cols:
            if col not in proba_df.columns:
                proba_df[col] = 0.0

        md = mahalanobis_distance_batch(x, baseline)
        ahi = calculate_ahi(md=md, b=float(self.config["health_index"]["b"]))
        rules = HealthRulesEngine(self.config["classification_thresholds"])
        health_class = rules.classify(ahi)

        result = df[["timestamp", "asset_id", *self.features]].copy()
        result["bp_prediction"] = bp_prediction
        result["mahalanobis_distance"] = md
        result["ahi"] = ahi
        result["health_classification"] = health_class
        result["maintenance_recommendation"] = result["health_classification"].map(recommendation_for_class)
        result = pd.concat([result, proba_df[expected_cols]], axis=1)

        result.to_csv(out_dir / "predictions.csv", index=False)
        result.to_json(out_dir / "predictions.json", orient="records", indent=2)

        trend = self._build_trend(result)
        trend.to_csv(out_dir / "health_trends.csv", index=False)

        powerbi = result.merge(trend, on=["asset_id", "timestamp"], how="left")
        powerbi.to_csv(out_dir / "powerbi_dataset.csv", index=False)

        with (out_dir / "model_evaluation.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        return {
            "quality_report": quality_report.to_dict(),
            "evaluation": metrics,
            "output_dir": str(out_dir),
        }

    @staticmethod
    def _build_trend(result: pd.DataFrame) -> pd.DataFrame:
        trend = result[["timestamp", "asset_id", "ahi"]].copy()
        trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
        trend = trend.dropna(subset=["timestamp"]).sort_values(["asset_id", "timestamp"])
        trend["ahi_trend_rolling3"] = trend.groupby("asset_id")["ahi"].transform(lambda s: s.rolling(3, min_periods=1).mean())
        trend["timestamp"] = trend["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return trend
