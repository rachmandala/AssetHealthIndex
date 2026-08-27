# Data Dictionary

This document describes every field used across the Asset Health Index (AHI)
platform's domain models and Power BI star-schema exports.

## Domain Models (`src/models/`)

### Asset

| Field | Type | Description |
|---|---|---|
| `asset_id` | string | Unique identifier of the asset (primary key). |
| `asset_name` | string | Human-readable asset name. |
| `asset_type` | string | Asset type/category (e.g., "Generator", "Turbine"). |
| `location` | string | Physical location or site name. |
| `commissioning_date` | date | Date the asset entered service. |
| `manufacturer` | string | Original equipment manufacturer. |
| `criticality` | enum (`Low`, `Medium`, `High`, `Critical`) | Business criticality ranking. |
| `status` | enum (`Active`, `Standby`, `Maintenance`, `Decommissioned`) | Current operational status. |

### OperationalMeasurement

| Field | Type | Description |
|---|---|---|
| `measurement_id` | string | Unique identifier of the measurement record. |
| `asset_id` | string | Foreign key reference to `Asset.asset_id`. |
| `timestamp` | datetime (UTC) | Time the measurement was captured. |
| `speed` | float | Rotational speed (RPM). |
| `frequency` | float | Electrical frequency (Hz). |
| `voltage` | float | Terminal voltage (V). |
| `current` | float | Line current (A). |
| `active_power` | float | Active (real) power output (MW). |
| `reactive_power` | float | Reactive power output (MVAr). |
| `power_factor` | float | Power factor, range [-1.0, 1.0]. |

### HealthAssessment

| Field | Type | Description |
|---|---|---|
| `assessment_id` | string | Unique identifier of the assessment record. |
| `asset_id` | string | Foreign key reference to `Asset.asset_id`. |
| `timestamp` | datetime (UTC) | Time the assessment corresponds to. |
| `bp_prediction` | enum (`Healthy`, `Sub-Healthy`, `Abnormal`, `Fault`) | BP Neural Network predicted condition class. |
| `bp_class_probabilities` | dict[str, float] (optional) | Per-class predicted probability. |
| `mahalanobis_distance` | float (≥ 0) | Mahalanobis Distance from the healthy baseline. |
| `health_index` | float [0, 1] | Health Index (HI). |
| `ahi_score` | float [0, 100] | Asset Health Index (AHI) = HI × 100. |
| `health_status` | enum (`Healthy`, `Sub-Healthy`, `Abnormal`, `Fault`) | Classified health status. |
| `recommendation` | string | Maintenance recommendation text. |
| `model_version` | string | Version identifier of the model(s) used. |

### ModelMetadata

| Field | Type | Description |
|---|---|---|
| `model_version` | string | Unique model version identifier. |
| `training_date` | date | Date the model was trained. |
| `parameters` | dict | Hyperparameters/configuration used for training. |
| `model_accuracy` | float [0, 1] | Accuracy (or primary metric) on the held-out test set. |
| `notes` | string (optional) | Free-text notes about this model version. |

## Power BI Star Schema (`src/exports/powerbi_export.py`)

### Fact_AssetHealthAssessment

| Column | Type | Source |
|---|---|---|
| `AssessmentId` | string | `HealthAssessment.assessment_id` |
| `Timestamp` | datetime | `HealthAssessment.timestamp` |
| `AssetId` | string | `HealthAssessment.asset_id` |
| `BPPrediction` | string | `HealthAssessment.bp_prediction` |
| `HealthyProbability` | float | `bp_class_probabilities["Healthy"]` |
| `SubHealthyProbability` | float | `bp_class_probabilities["Sub-Healthy"]` |
| `AbnormalProbability` | float | `bp_class_probabilities["Abnormal"]` |
| `FaultProbability` | float | `bp_class_probabilities["Fault"]` |
| `MahalanobisDistance` | float | `HealthAssessment.mahalanobis_distance` |
| `HealthIndex` | float | `HealthAssessment.health_index` |
| `AHIScore` | float | `HealthAssessment.ahi_score` |
| `HealthStatus` | string | `HealthAssessment.health_status` |
| `Recommendation` | string | `HealthAssessment.recommendation` |
| `ModelVersion` | string | `HealthAssessment.model_version` |

### Dim_Asset

| Column | Type | Source |
|---|---|---|
| `AssetId` | string | `Asset.asset_id` |
| `AssetName` | string | `Asset.asset_name` |
| `AssetType` | string | `Asset.asset_type` |
| `Location` | string | `Asset.location` |
| `CommissioningDate` | date | `Asset.commissioning_date` |
| `Manufacturer` | string | `Asset.manufacturer` |
| `Criticality` | string | `Asset.criticality` |
| `Status` | string | `Asset.status` |

### Dim_Date

| Column | Type | Description |
|---|---|---|
| `DateKey` | int (yyyymmdd) | Surrogate key for the date. |
| `Date` | date | Calendar date. |
| `Year` | int | Calendar year. |
| `Quarter` | int | Calendar quarter (1-4). |
| `Month` | int | Calendar month (1-12). |
| `MonthName` | string | Full month name. |
| `Week` | int | ISO calendar week number. |
| `Day` | int | Day of month. |
| `DayName` | string | Full day-of-week name. |

### Dim_HealthStatus

| Column | Type | Description |
|---|---|---|
| `StatusKey` | int | Surrogate key. |
| `HealthStatus` | string | Health status label. |
| `Description` | string | Business-friendly description of the status. |
| `ColorCode` | string (hex) | Dashboard color code (Green/Yellow/Orange/Red). |

### Dim_ModelVersion

| Column | Type | Description |
|---|---|---|
| `ModelVersion` | string | Unique model version identifier. |
| `VersionName` | string | Display name for the model version. |
| `TrainingDate` | date | Date the model was trained. |
| `Algorithm` | string | Algorithm description. |
| `Notes` | string | Free-text notes. |

## Configuration Fields (`config/config.yaml`)

| Section | Field | Description |
|---|---|---|
| `model` | `train_split` / `test_split` | Stratified train/test split ratios (default 0.60 / 0.40). |
| `bp_network` | `hidden_layer_sizes`, `activation`, `solver`, `alpha`, `learning_rate_init`, `max_iter`, `random_state` | `MLPClassifier` hyperparameters. |
| `mahalanobis` | `covariance_regularization` | Regularization term added to the covariance diagonal. |
| `health_index` | `b` | Sensitivity parameter in the HI formula. |
| `thresholds` | `healthy`, `subhealthy`, `abnormal` | AHI classification threshold values. |
| `validation.ranges` | `<feature>.min` / `<feature>.max` | Plausible value ranges per input feature. |
| `preprocessing` | `outlier_method`, `outlier_iqr_multiplier`, `missing_value_strategy` | Preprocessing strategy configuration. |
