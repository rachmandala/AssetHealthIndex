"""Power BI star-schema dataset export layer.

Builds and exports the fact and dimension tables that power the AHI
Power BI dashboards (Executive, Fleet Health, Asset Health, Risk, and
Model Monitoring). All exports are CSV files with business-friendly
column names and explicit datatypes, ready for direct import into Power
BI or Microsoft Fabric.

Star schema:

    Fact_AssetHealthAssessment  (grain: one row per assessment)
        -> Dim_Asset            (asset_id)
        -> Dim_Date             (date_key)
        -> Dim_HealthStatus     (status_key)
        -> Dim_ModelVersion     (model_version)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.settings import AppSettings, get_settings
from src.models.asset import Asset
from src.models.health_assessment import HealthAssessment
from src.models.model_metadata import ModelMetadata
from src.utils.constants import HEALTH_STATUS_COLOR_CODES, HEALTH_STATUS_DESCRIPTIONS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PowerBIExporter:
    """Builds and writes Power BI-ready star-schema CSV datasets."""

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Fact table
    # ------------------------------------------------------------------
    def build_fact_asset_health_assessment(
        self, assessments: list[HealthAssessment]
    ) -> pd.DataFrame:
        """Build the ``Fact_AssetHealthAssessment`` fact table.

        Args:
            assessments: Health assessment records to export.

        Returns:
            A DataFrame with one row per assessment and business-friendly
            column names matching the star-schema contract.
        """
        records = []
        for a in assessments:
            probs = a.bp_class_probabilities or {}
            records.append(
                {
                    "AssessmentId": a.assessment_id,
                    "Timestamp": a.timestamp,
                    "AssetId": a.asset_id,
                    "BPPrediction": str(a.bp_prediction),
                    "HealthyProbability": probs.get("Healthy"),
                    "SubHealthyProbability": probs.get("Sub-Healthy"),
                    "AbnormalProbability": probs.get("Abnormal"),
                    "FaultProbability": probs.get("Fault"),
                    "MahalanobisDistance": a.mahalanobis_distance,
                    "HealthIndex": a.health_index,
                    "AHIScore": a.ahi_score,
                    "HealthStatus": str(a.health_status),
                    "Recommendation": a.recommendation,
                    "ModelVersion": a.model_version,
                }
            )
        df = pd.DataFrame.from_records(records)
        if not df.empty:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        return df

    # ------------------------------------------------------------------
    # Dimension tables
    # ------------------------------------------------------------------
    def build_dim_asset(self, assets: list[Asset]) -> pd.DataFrame:
        """Build the ``Dim_Asset`` dimension table."""
        records = [
            {
                "AssetId": a.asset_id,
                "AssetName": a.asset_name,
                "AssetType": a.asset_type,
                "Location": a.location,
                "CommissioningDate": a.commissioning_date,
                "Manufacturer": a.manufacturer,
                "Criticality": str(a.criticality),
                "Status": str(a.status),
            }
            for a in assets
        ]
        df = pd.DataFrame.from_records(records)
        if not df.empty:
            df["CommissioningDate"] = pd.to_datetime(df["CommissioningDate"])
        return df

    def build_dim_date(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Build the ``Dim_Date`` dimension table spanning a date range.

        Args:
            start_date: First calendar date to include (inclusive).
            end_date: Last calendar date to include (inclusive).

        Returns:
            A DataFrame with one row per calendar date in the range.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        df = pd.DataFrame(
            {
                "DateKey": dates.strftime("%Y%m%d").astype(int),
                "Date": dates,
                "Year": dates.year,
                "Quarter": dates.quarter,
                "Month": dates.month,
                "MonthName": dates.strftime("%B"),
                "Week": dates.isocalendar().week.values,
                "Day": dates.day,
                "DayName": dates.strftime("%A"),
            }
        )
        return df

    def build_dim_health_status(self) -> pd.DataFrame:
        """Build the ``Dim_HealthStatus`` dimension table.

        Status ordering/keys and color codes are derived from
        ``src.utils.constants`` so dashboard color mapping stays
        consistent with the classification engine.
        """
        statuses = list(HEALTH_STATUS_COLOR_CODES.keys())
        df = pd.DataFrame(
            {
                "StatusKey": range(1, len(statuses) + 1),
                "HealthStatus": statuses,
                "Description": [HEALTH_STATUS_DESCRIPTIONS[s] for s in statuses],
                "ColorCode": [HEALTH_STATUS_COLOR_CODES[s] for s in statuses],
            }
        )
        return df

    def build_dim_model_version(self, model_metadata: list[ModelMetadata]) -> pd.DataFrame:
        """Build the ``Dim_ModelVersion`` dimension table."""
        records = [
            {
                "ModelVersion": m.model_version,
                "VersionName": m.model_version,
                "TrainingDate": m.training_date,
                "Algorithm": "BP Neural Network + Mahalanobis Distance",
                "Notes": m.notes or "",
            }
            for m in model_metadata
        ]
        df = pd.DataFrame.from_records(records)
        if not df.empty:
            df["TrainingDate"] = pd.to_datetime(df["TrainingDate"])
        return df

    # ------------------------------------------------------------------
    # File export
    # ------------------------------------------------------------------
    def export_all(
        self,
        assessments: list[HealthAssessment],
        assets: list[Asset],
        model_metadata: list[ModelMetadata],
        output_dir: Optional[str | Path] = None,
    ) -> dict[str, Path]:
        """Build and write all star-schema CSV exports.

        Args:
            assessments: Health assessment records.
            assets: Asset master data records.
            model_metadata: Registered model version metadata.
            output_dir: Destination directory; defaults to configured
                ``paths.exports_dir``.

        Returns:
            A mapping of logical dataset name to the written file path.
        """
        output_dir = Path(output_dir) if output_dir else self._settings.resolve_path(
            self._settings.paths.exports_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        fact_df = self.build_fact_asset_health_assessment(assessments)
        dim_asset_df = self.build_dim_asset(assets)
        dim_status_df = self.build_dim_health_status()
        dim_model_df = self.build_dim_model_version(model_metadata)

        if not fact_df.empty:
            start_date = fact_df["Timestamp"].min().normalize()
            end_date = fact_df["Timestamp"].max().normalize()
        else:
            start_date = end_date = pd.Timestamp.utcnow().normalize()
        dim_date_df = self.build_dim_date(start_date, end_date)

        exports_cfg = self._settings.exports
        written_paths: dict[str, Path] = {}

        file_map = {
            "fact_asset_health_assessment": (fact_df, exports_cfg.fact_asset_health_assessment),
            "dim_asset": (dim_asset_df, exports_cfg.dim_asset),
            "dim_date": (dim_date_df, exports_cfg.dim_date),
            "dim_health_status": (dim_status_df, exports_cfg.dim_health_status),
            "dim_model_version": (dim_model_df, exports_cfg.dim_model_version),
        }

        for name, (df, filename) in file_map.items():
            path = output_dir / filename
            df.to_csv(path, index=False)
            written_paths[name] = path
            logger.info("Exported '%s' (%d rows) to '%s'.", name, len(df), path)

        return written_paths
