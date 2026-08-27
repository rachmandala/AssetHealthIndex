"""Phase 2 - Step 1: Data validation.

Validates the raw source datasets (file existence, schema, missing values,
duplicates, invalid data types, invalid labels, and plausible value ranges)
and writes a consolidated JSON validation report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.pipeline.common import (
    DIM_ASSET_PATH,
    NEW_MEASUREMENTS_PATH,
    OPERATIONAL_MEASUREMENTS_PATH,
    VALIDATION_REPORT_PATH,
)
from src.utils.logger import get_logger
from src.validation.validator import (
    DataValidationError,
    OperationalMeasurementValidator,
    check_file_exists,
)

logger = get_logger(__name__)

DIM_ASSET_REQUIRED_COLUMNS = (
    "asset_id",
    "asset_name",
    "asset_type",
    "location",
    "manufacturer",
    "criticality",
    "status",
    "commissioning_date",
)


def validate_dim_asset(path: Path = DIM_ASSET_PATH) -> dict:
    """Validate the asset master data file (existence + required columns)."""
    resolved = check_file_exists(path)
    df = pd.read_csv(resolved)
    missing_columns = [c for c in DIM_ASSET_REQUIRED_COLUMNS if c not in df.columns]
    duplicate_ids = df["asset_id"].duplicated().sum() if "asset_id" in df.columns else 0

    is_valid = not missing_columns and duplicate_ids == 0
    result = {
        "total_records": len(df),
        "is_valid": is_valid,
        "missing_columns": missing_columns,
        "duplicate_asset_ids": int(duplicate_ids),
    }
    logger.info("[dim_asset.csv] %s", result)
    return result


def run_validation(raise_on_error: bool = True) -> dict:
    """Validate all three raw datasets and write the consolidated report.

    Args:
        raise_on_error: Raise :class:`DataValidationError` if the historical
            training dataset (``operational_measurements.csv``) fails
            validation. New measurements only ever produce warnings/errors
            that are reported, since scoring can proceed per-row.

    Returns:
        The consolidated validation report as a dictionary (also written to
        ``data/processed/validation_report.json``).
    """
    validator = OperationalMeasurementValidator()

    dim_asset_result = validate_dim_asset()

    _, operational_report = validator.validate_raw_dataset(
        OPERATIONAL_MEASUREMENTS_PATH, require_label=True, raise_on_error=False
    )
    _, new_measurements_report = validator.validate_raw_dataset(
        NEW_MEASUREMENTS_PATH, require_label=False, raise_on_error=False
    )

    consolidated = {
        "dim_asset": dim_asset_result,
        "operational_measurements": operational_report.to_dict(),
        "new_measurements": new_measurements_report.to_dict(),
    }

    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, default=str)
    logger.info("Wrote validation report to '%s'.", VALIDATION_REPORT_PATH)

    if raise_on_error and not operational_report.is_valid:
        raise DataValidationError(
            "operational_measurements.csv failed validation with "
            f"{operational_report.error_count} error(s). "
            f"See {VALIDATION_REPORT_PATH} for details."
        )

    return consolidated


if __name__ == "__main__":
    run_validation()
