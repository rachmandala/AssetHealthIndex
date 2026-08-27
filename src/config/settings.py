"""Application settings loaded from ``config/config.yaml``.

Exposes strongly-typed, validated configuration objects using Pydantic so
the rest of the application never reads raw YAML dictionaries directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"


class ProjectSettings(BaseModel):
    """General project metadata."""

    name: str
    description: str = ""
    version: str = "0.1.0"
    environment: str = "development"


class PathSettings(BaseModel):
    """Filesystem paths used throughout the platform."""

    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    exports_dir: str = "data/exports"
    artifacts_dir: str = "artifacts"


class ModelTrainingSettings(BaseModel):
    """Train/test split configuration for model training."""

    train_split: float = Field(default=0.60, gt=0.0, lt=1.0)
    test_split: float = Field(default=0.40, gt=0.0, lt=1.0)
    stratify: bool = True
    random_state: int = 42


class BPNetworkSettings(BaseModel):
    """Hyperparameters for the BP Neural Network (MLPClassifier)."""

    hidden_layer_sizes: list[int] = Field(default_factory=lambda: [20])
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0001
    learning_rate_init: float = 0.001
    max_iter: int = 1000
    random_state: int = 42
    early_stopping: bool = False
    classes: list[str] = Field(
        default_factory=lambda: ["Healthy", "Sub-Healthy", "Abnormal", "Fault"]
    )


class MahalanobisSettings(BaseModel):
    """Configuration for the Mahalanobis Distance health model."""

    covariance_regularization: float = 0.0001
    input_features: list[str] = Field(
        default_factory=lambda: [
            "speed",
            "frequency",
            "voltage",
            "current",
            "active_power",
            "reactive_power",
            "power_factor",
        ]
    )


class HealthIndexSettings(BaseModel):
    """Configuration for the Health Index (HI) formula."""

    b: float = 0.5


class ThresholdSettings(BaseModel):
    """AHI classification thresholds (upper-exclusive bounds)."""

    healthy: float = 90
    subhealthy: float = 60
    abnormal: float = 30


class ValidationRangeSettings(BaseModel):
    """Min/max plausible range for one operational measurement field."""

    min: float
    max: float


class ValidationSettings(BaseModel):
    """Data validation configuration."""

    ranges: dict[str, ValidationRangeSettings] = Field(default_factory=dict)


class PreprocessingSettings(BaseModel):
    """Preprocessing configuration."""

    outlier_method: str = "iqr"
    outlier_iqr_multiplier: float = 1.5
    missing_value_strategy: str = "median"
    scaler_artifact_name: str = "feature_scaler.joblib"


class ExportSettings(BaseModel):
    """Power BI export file names."""

    fact_asset_health_assessment: str = "fact_asset_health_assessment.csv"
    dim_asset: str = "dim_asset.csv"
    dim_date: str = "dim_date.csv"
    dim_health_status: str = "dim_health_status.csv"
    dim_model_version: str = "dim_model_version.csv"


class AppSettings(BaseModel):
    """Root application settings object composed of all sub-sections."""

    project: ProjectSettings
    paths: PathSettings = Field(default_factory=PathSettings)
    model: ModelTrainingSettings = Field(default_factory=ModelTrainingSettings)
    bp_network: BPNetworkSettings = Field(default_factory=BPNetworkSettings)
    mahalanobis: MahalanobisSettings = Field(default_factory=MahalanobisSettings)
    health_index: HealthIndexSettings = Field(default_factory=HealthIndexSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    exports: ExportSettings = Field(default_factory=ExportSettings)

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a configured relative path against the repository root."""
        return _REPO_ROOT / relative_path


def _load_yaml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file is not a valid mapping: {config_path}")
    return data


@lru_cache(maxsize=1)
def get_settings(config_path: Optional[str] = None) -> AppSettings:
    """Load and cache application settings from YAML.

    Args:
        config_path: Optional override path to a config YAML file. When
            omitted, ``config/config.yaml`` at the repository root is used.

    Returns:
        A validated :class:`AppSettings` instance.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    raw = _load_yaml(path)
    return AppSettings(**raw)


def reload_settings(config_path: Optional[str] = None) -> AppSettings:
    """Clear the settings cache and reload from disk.

    Useful in tests or when the configuration file changes at runtime.
    """
    get_settings.cache_clear()
    return get_settings(config_path)
