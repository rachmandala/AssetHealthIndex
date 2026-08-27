# Methodology

This document explains the Asset Health Index (AHI) methodology implemented
by this platform, based on **Han et al. (2023)**.

## 1. BP Neural Network Methodology

A Back Propagation (BP) Neural Network is a supervised, feed-forward
multilayer perceptron trained via the back-propagation algorithm. In this
platform it is implemented using `sklearn.neural_network.MLPClassifier`
(`src/ml/bp_network.py`).

**Inputs (7 variables):**

1. Speed
2. Frequency
3. Voltage
4. Current
5. Active Power
6. Reactive Power
7. Power Factor

**Outputs (4 condition classes):**

- Healthy
- Sub-Healthy
- Abnormal
- Fault

The network is trained on standardized (zero mean, unit variance) feature
data using a stratified train/test split (default 60% train / 40% test) so
that the class distribution is preserved across both sets. Hyperparameters
(hidden layer sizes, activation, solver, learning rate, iterations, random
state) are configurable via `config/config.yaml` under `bp_network`.

The BP Neural Network's role in the pipeline is **condition classification**:
producing a categorical label (and class probabilities) that complements the
continuous Mahalanobis-Distance-based health score.

## 2. Healthy Baseline Creation

The Mahalanobis Distance approach requires a reference "healthy" operating
baseline, characterized by:

- **Mean vector (μ)**: the average of each of the 7 input features across a
  representative set of observations known to be in a healthy condition
  (e.g., BP Neural Network label = "Healthy", or engineering-defined normal
  operating periods).
- **Covariance matrix (Σ)**: the covariance between the 7 input features
  across that same healthy reference set, capturing normal joint variability
  and correlation between variables (e.g., voltage and current naturally
  co-varying under healthy operation).

This baseline is fit once (per asset or per asset class, depending on
deployment strategy) via `MahalanobisHealthModel.fit_healthy_baseline()` and
persisted with `joblib` for reuse at inference time.

## 3. Mahalanobis Distance Methodology

The Mahalanobis Distance (MD) measures how many "standard deviations" a new
observation `X` is from the healthy baseline, while accounting for
correlations between variables (unlike simple Euclidean distance).

**Formula:**

```
MD = sqrt( (X - μ)ᵀ · Σ⁻¹ · (X - μ) )
```

Where:

- `X` = current standardized observation (7-element feature vector)
- `μ` = healthy baseline mean vector
- `Σ⁻¹` = inverse of the healthy baseline covariance matrix

**Covariance regularization:** to guarantee that `Σ` is invertible (especially
when the healthy baseline sample size is limited relative to the number of
features, or features are highly correlated), a small regularization term is
added to the diagonal before inversion:

```
Σ_regularized = Σ + λ · I
```

Where `λ` is `mahalanobis.covariance_regularization` in configuration
(default `0.0001`) and `I` is the identity matrix.

A larger MD indicates the observation deviates more significantly from
normal healthy operation across the combined 7-variable feature space.

## 4. Health Index (HI) Formula

The Health Index converts an unbounded, non-negative Mahalanobis Distance
into a bounded, interpretable score in the range `(0, 1]`:

```
HI = 1 - (2 / π) · arctan(b · MD)
```

Where `b` is a configurable sensitivity parameter
(`health_index.b` in `config/config.yaml`, default `0.5`). Larger values of
`b` make HI more sensitive to (i.e., decay faster with) increasing MD.

Properties:

- When `MD = 0` (observation exactly matches the healthy baseline), `HI = 1`.
- As `MD → ∞`, `HI → 0`.
- `HI` is monotonically decreasing in `MD`.

## 5. Asset Health Index (AHI) Calculation Logic

The Asset Health Index rescales HI onto a familiar 0-100 scale:

```
AHI = HI × 100
```

**Classification thresholds** (configurable via `config/config.yaml` under
`thresholds`):

| Condition | Health Status |
|---|---|
| AHI > 90 | Healthy |
| 60 < AHI ≤ 90 | Sub-Healthy |
| 30 < AHI ≤ 60 | Abnormal |
| AHI ≤ 30 | Fault |

**Recommendation mapping:**

| Health Status | Recommendation |
|---|---|
| Healthy | Continue Normal Operation |
| Sub-Healthy | Increase Monitoring Frequency |
| Abnormal | Schedule Inspection |
| Fault | Immediate Maintenance Assessment |

## End-to-End Flow Summary

1. Ingest an operational measurement (7 input variables) for an asset.
2. Validate and preprocess (missing values, outliers, standardization).
3. Run the BP Neural Network to classify the condition (Healthy /
   Sub-Healthy / Abnormal / Fault) and obtain class probabilities.
4. Calculate the Mahalanobis Distance of the standardized observation from
   the asset's healthy baseline.
5. Convert MD to a Health Index (HI) and Asset Health Index (AHI) score.
6. Classify the AHI score into a health status band.
7. Generate a maintenance recommendation.
8. Persist the full assessment and export it into the Power BI star schema.
