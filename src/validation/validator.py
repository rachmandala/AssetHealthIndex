"""Data validation and data quality checks for operational measurements.

Implements schema validation, missing value detection, duplicate record
detection, invalid data type detection, and range checking, along with
structured data quality reporting and logging.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.config.settings import AppSettings, get_settings
from src.utils.constants import BP_CONDITION_CLASSES, INPUT_FEATURE_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "measurement_id",
    "asset_id",
    "timestamp",
    *INPUT_FEATURE_NAMES,
)

# Columns present in the raw CSV datasets (no synthetic measurement_id).
RAW_BASE_COLUMNS: tuple[str, ...] = ("timestamp", "asset_id", *INPUT_FEATURE_NAMES)
RAW_LABELED_COLUMNS: tuple[str, ...] = (*RAW_BASE_COLUMNS, "label")

ALLOWED_LABELS: tuple[str, ...] = BP_CONDITION_CLASSES


def check_file_exists(path: str | Path) -> Path:
    """Verify a source data file exists on disk.

    Args:
        path: Path to the expected input file.

    Returns:
        The resolved :class:`Path`.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Required input file not found: {resolved}")
    return resolved


class DataValidationError(Exception):
    """Raised when input data fails validation and cannot be processed."""


@dataclass
class ValidationIssue:
    """A single validation finding for a specific record/field."""

    severity: str  # "error" | "warning"
    check: str
    message: str
    row_index: Any = None
    column: str | None = None


@dataclass
class ValidationReport:
    """Aggregated result of running the validator against a dataset.

    Attributes:
        total_records: Number of records evaluated.
        issues: List of individual validation findings.
        is_valid: ``True`` when no error-level issues were found.
    """

    total_records: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def summary(self) -> str:
        """Return a human-readable summary line for logging/reporting."""
        return (
            f"ValidationReport(total_records={self.total_records}, "
            f"errors={self.error_count}, warnings={self.warning_count}, "
            f"is_valid={self.is_valid})"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the issues as a tabular DataFrame for reporting/export."""
        if not self.issues:
            return pd.DataFrame(
                columns=["severity", "check", "message", "row_index", "column"]
            )
        return pd.DataFrame([issue.__dict__ for issue in self.issues])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this report."""
        return {
            "total_records": self.total_records,
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [asdict(issue) for issue in self.issues],
        }

    def save_json(self, path: str | Path) -> Path:
        """Write this report as a JSON file and return the written path."""
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return resolved


class OperationalMeasurementValidator:
    """Validates raw operational measurement data before preprocessing.

    Performs, in order:
        1. Schema validation (required columns present).
        2. Invalid data type detection (numeric fields must be numeric).
        3. Missing value detection.
        4. Duplicate record detection (by ``measurement_id``).
        5. Range checking against configured plausible bounds.
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()

    def validate_schema(
        self, df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS
    ) -> list[ValidationIssue]:
        """Check that all required columns are present in the DataFrame."""
        issues: list[ValidationIssue] = []
        missing_columns = [col for col in required_columns if col not in df.columns]
        for col in missing_columns:
            issues.append(
                ValidationIssue(
                    severity="error",
                    check="schema",
                    message=f"Required column '{col}' is missing from input data.",
                    column=col,
                )
            )
        return issues

    def check_missing_values(
        self, df: pd.DataFrame, columns: Iterable[str] = INPUT_FEATURE_NAMES
    ) -> list[ValidationIssue]:
        """Detect missing (NaN/None) values in the specified columns."""
        issues: list[ValidationIssue] = []
        for col in columns:
            if col not in df.columns:
                continue
            missing_rows = df.index[df[col].isna()].tolist()
            for row_index in missing_rows:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        check="missing_value",
                        message=f"Missing value in column '{col}'.",
                        row_index=row_index,
                        column=col,
                    )
                )
        return issues

    def check_duplicates(
        self, df: pd.DataFrame, key_column: str = "measurement_id"
    ) -> list[ValidationIssue]:
        """Detect duplicate records based on a unique key column."""
        issues: list[ValidationIssue] = []
        if key_column not in df.columns:
            return issues
        duplicated_mask = df.duplicated(subset=[key_column], keep="first")
        for row_index in df.index[duplicated_mask].tolist():
            issues.append(
                ValidationIssue(
                    severity="error",
                    check="duplicate_record",
                    message=(
                        f"Duplicate record for '{key_column}' = "
                        f"'{df.loc[row_index, key_column]}'."
                    ),
                    row_index=row_index,
                    column=key_column,
                )
            )
        return issues

    def check_duplicates_by_keys(
        self, df: pd.DataFrame, key_columns: Iterable[str] = ("asset_id", "timestamp")
    ) -> list[ValidationIssue]:
        """Detect duplicate records based on a composite key.

        Used for the raw CSV datasets, which identify a record by
        ``(asset_id, timestamp)`` rather than a synthetic ``measurement_id``.
        """
        key_columns = [c for c in key_columns if c in df.columns]
        issues: list[ValidationIssue] = []
        if not key_columns:
            return issues
        duplicated_mask = df.duplicated(subset=key_columns, keep="first")
        for row_index in df.index[duplicated_mask].tolist():
            key_values = tuple(df.loc[row_index, key_columns])
            issues.append(
                ValidationIssue(
                    severity="error",
                    check="duplicate_record",
                    message=(
                        f"Duplicate record for {tuple(key_columns)} = {key_values}."
                    ),
                    row_index=row_index,
                    column=",".join(key_columns),
                )
            )
        return issues

    def check_labels(
        self,
        df: pd.DataFrame,
        label_column: str = "label",
        allowed_labels: Iterable[str] = ALLOWED_LABELS,
    ) -> list[ValidationIssue]:
        """Detect label values outside the allowed BP condition classes."""
        issues: list[ValidationIssue] = []
        if label_column not in df.columns:
            return issues
        allowed = set(allowed_labels)
        invalid_mask = ~df[label_column].isin(allowed)
        for row_index in df.index[invalid_mask].tolist():
            issues.append(
                ValidationIssue(
                    severity="error",
                    check="invalid_label",
                    message=(
                        f"Invalid label '{df.loc[row_index, label_column]}'. "
                        f"Expected one of {sorted(allowed)}."
                    ),
                    row_index=row_index,
                    column=label_column,
                )
            )
        return issues

    def check_data_types(
        self, df: pd.DataFrame, numeric_columns: Iterable[str] = INPUT_FEATURE_NAMES
    ) -> list[ValidationIssue]:
        """Detect non-numeric values in columns expected to be numeric."""
        issues: list[ValidationIssue] = []
        for col in numeric_columns:
            if col not in df.columns:
                continue
            coerced = pd.to_numeric(df[col], errors="coerce")
            invalid_mask = coerced.isna() & df[col].notna()
            for row_index in df.index[invalid_mask].tolist():
                issues.append(
                    ValidationIssue(
                        severity="error",
                        check="invalid_data_type",
                        message=(
                            f"Non-numeric value '{df.loc[row_index, col]}' found in "
                            f"numeric column '{col}'."
                        ),
                        row_index=row_index,
                        column=col,
                    )
                )
        return issues

    def check_ranges(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Detect values outside the configured plausible min/max range."""
        issues: list[ValidationIssue] = []
        ranges = self._settings.validation.ranges
        for col, bounds in ranges.items():
            if col not in df.columns:
                continue
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            out_of_range_mask = (numeric_col < bounds.min) | (numeric_col > bounds.max)
            for row_index in df.index[out_of_range_mask.fillna(False)].tolist():
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        check="range_check",
                        message=(
                            f"Value '{df.loc[row_index, col]}' in column '{col}' is "
                            f"outside expected range [{bounds.min}, {bounds.max}]."
                        ),
                        row_index=row_index,
                        column=col,
                    )
                )
        return issues

    def validate(
        self, df: pd.DataFrame, raise_on_error: bool = False
    ) -> ValidationReport:
        """Run the full validation suite against a DataFrame of measurements.

        Args:
            df: Raw operational measurement records.
            raise_on_error: When ``True``, raises :class:`DataValidationError`
                if any error-level issue is found.

        Returns:
            A :class:`ValidationReport` describing all findings.

        Raises:
            DataValidationError: If ``raise_on_error`` is ``True`` and the
                report contains at least one error-level issue.
        """
        issues: list[ValidationIssue] = []
        issues.extend(self.validate_schema(df))

        # Only run row-level checks if the schema is at least partially valid.
        issues.extend(self.check_data_types(df))
        issues.extend(self.check_missing_values(df))
        issues.extend(self.check_duplicates(df))
        issues.extend(self.check_ranges(df))

        report = ValidationReport(total_records=len(df), issues=issues)
        logger.info(report.summary())

        for issue in issues:
            log_fn = logger.error if issue.severity == "error" else logger.warning
            log_fn(
                "[%s] %s (row=%s, column=%s)",
                issue.check,
                issue.message,
                issue.row_index,
                issue.column,
            )

        if raise_on_error and not report.is_valid:
            raise DataValidationError(
                f"Validation failed with {report.error_count} error(s). "
                f"See ValidationReport.issues for details."
            )

        return report

    def validate_raw_dataset(
        self,
        source_path: str | Path,
        require_label: bool,
        raise_on_error: bool = False,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Validate one of the raw CSV datasets end-to-end.

        Performs, in order: file existence, schema validation, invalid data
        type detection, missing value detection, duplicate detection (by
        ``asset_id`` + ``timestamp``), range checking, and (when
        ``require_label`` is ``True``) invalid label detection.

        Args:
            source_path: Path to the raw CSV file
                (``operational_measurements.csv`` or ``new_measurements.csv``).
            require_label: Whether a ``label`` column is required (``True``
                for historical training data, ``False`` for new/unlabeled
                measurements awaiting scoring).
            raise_on_error: Raise :class:`DataValidationError` if any
                error-level issue is found.

        Returns:
            A tuple of ``(dataframe, validation_report)``.
        """
        path = check_file_exists(source_path)
        df = pd.read_csv(path)

        required_columns = RAW_LABELED_COLUMNS if require_label else RAW_BASE_COLUMNS
        issues: list[ValidationIssue] = []
        issues.extend(self.validate_schema(df, required_columns=required_columns))
        issues.extend(self.check_data_types(df))
        issues.extend(self.check_missing_values(df))
        issues.extend(self.check_duplicates_by_keys(df))
        issues.extend(self.check_ranges(df))
        if require_label:
            issues.extend(self.check_labels(df))

        report = ValidationReport(total_records=len(df), issues=issues)
        logger.info("[%s] %s", path.name, report.summary())
        for issue in issues:
            log_fn = logger.error if issue.severity == "error" else logger.warning
            log_fn(
                "[%s] [%s] %s (row=%s, column=%s)",
                path.name,
                issue.check,
                issue.message,
                issue.row_index,
                issue.column,
            )

        if raise_on_error and not report.is_valid:
            raise DataValidationError(
                f"Validation of '{path}' failed with {report.error_count} "
                f"error(s). See ValidationReport.issues for details."
            )

        return df, report


def validate_timestamp_not_future(
    timestamp: datetime, reference: datetime | None = None
) -> bool:
    """Return ``True`` if ``timestamp`` is not in the future relative to ``reference``."""
    reference = reference or datetime.utcnow()
    return timestamp <= reference
