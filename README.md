# Asset Health Index (AHI) Platform

Power generation asset health index model, based on the methodology described
by **Han et al. (2023)**, combining a **BP Neural Network**, **Mahalanobis
Distance (MD)**, and a **Health Index (HI) / Asset Health Index (AHI)**
scoring system.

> **Status:** Phase 1 — repository foundation, architecture, configuration,
> documentation, data models, interfaces, and core algorithm modules.
> Real data ingestion and model training will be implemented in later phases.

## Project Overview

The AHI platform continuously evaluates the health condition of power
generation assets (generators, turbines, etc.) using operational
measurements, and produces:

- A **condition classification** (Healthy / Sub-Healthy / Abnormal / Fault)
  from a BP Neural Network.
- A continuous **Mahalanobis Distance** from a healthy operating baseline.
- A bounded **Health Index (HI)** and **Asset Health Index (AHI)** score
  (0-100 scale).
- A **health status classification** and **maintenance recommendation**.
- **Power BI-ready** star-schema datasets for dashboarding.

### Target Users

- Reliability Engineers
- Asset Management Engineers
- Maintenance Engineers
- Plant Performance Engineers
- Operations Engineers

## Han et al. (2023) Methodology

**Input variables (7):** Speed, Frequency, Voltage, Current, Active Power,
Reactive Power, Power Factor.

**BP Neural Network output classes:** Healthy, Sub-Healthy, Abnormal, Fault.

**Mahalanobis Distance:**

```
MD = sqrt((X - μ)ᵀ Σ⁻¹ (X - μ))
```

**Health Index:**

```
HI = 1 - (2 / π) × arctan(b × MD)
AHI = HI × 100
```

**Health classification (default thresholds):**

| Condition | Health Status | Recommendation |
|---|---|---|
| AHI > 90 | Healthy | Continue Normal Operation |
| 60 < AHI ≤ 90 | Sub-Healthy | Increase Monitoring Frequency |
| 30 < AHI ≤ 60 | Abnormal | Schedule Inspection |
| AHI ≤ 30 | Fault | Immediate Maintenance Assessment |

See [docs/methodology.md](docs/methodology.md) for the full explanation and
[docs/architecture.md](docs/architecture.md) for the system architecture and
Mermaid diagrams.

## Repository Structure

```
asset-health-index/
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── config/
│   ├── config.yaml
│   └── logging.yaml
├── artifacts/
├── data/
│   ├── raw/
│   ├── processed/
│   └── exports/
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   └── data_dictionary.md
├── notebooks/
├── src/
│   ├── main.py
│   ├── config/settings.py
│   ├── models/            # Pydantic domain models
│   ├── repositories/      # Interfaces + mock (in-memory) implementations
│   ├── validation/        # Data validation & quality checks
│   ├── preprocessing/     # Missing value/outlier handling, standardization
│   ├── ml/                # BP Neural Network, Mahalanobis, Health Index, training
│   ├── rules/             # Health classification engine
│   ├── recommendations/   # Maintenance recommendation engine
│   ├── exports/           # Power BI star-schema export layer
│   └── utils/             # Logging & shared constants
└── tests/                 # pytest unit tests
```

## Installation

### Prerequisites

- Python 3.11+
- Git

### Setup

```powershell
git clone <repository-url>
cd asset-health-index
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run tests

```powershell
pytest
```

### Run the application skeleton

```powershell
python -m src.main
```

## VS Code Setup

1. Open the repository folder in VS Code.
2. Install the **Python** extension (`ms-python.python`).
3. Select the `.venv` interpreter via **Python: Select Interpreter**.
4. Run tests via the **Testing** panel (pytest auto-discovery is configured
   in `pytest.ini`).

## GitHub Codespaces Setup

This repository can be opened directly in **GitHub Codespaces**:

1. On GitHub, click **Code → Codespaces → Create codespace on main**.
2. Once the Codespace loads, run:
   ```bash
   pip install -r requirements.txt
   pytest
   ```
3. Develop using the same VS Code experience in the browser or via the
   Codespaces desktop connection.

## Configuration

All model hyperparameters, thresholds, and business rules are externalized to
[config/config.yaml](config/config.yaml):

```yaml
project:
  name: Asset Health Index

model:
  train_split: 0.60
  test_split: 0.40

bp_network:
  hidden_layer_sizes: [20]
  max_iter: 1000
  random_state: 42

health_index:
  b: 0.5

thresholds:
  healthy: 90
  subhealthy: 60
  abnormal: 30
```

Logging is configured via [config/logging.yaml](config/logging.yaml).

## Future Roadmap

- **Phase 2**: Dummy/synthetic data generation for development and demos.
- **Phase 3**: Real database integration (SQL repository implementations).
- **Phase 4**: Model training with real operational data, model registry.
- **Phase 5**: Deployment to Azure ML / Microsoft Fabric for managed training
  and inference pipelines.
- **Phase 6**: Power BI Service dataset publishing and dashboard delivery.
- **Phase 7**: Copilot Studio integration for conversational maintenance
  recommendations.

## Documentation

- [docs/architecture.md](docs/architecture.md) — System architecture and diagrams.
- [docs/methodology.md](docs/methodology.md) — Han et al. (2023) methodology explained.
- [docs/data_dictionary.md](docs/data_dictionary.md) — Full field-level data dictionary.

## References

Han, et al. (2023). *Asset Health Index methodology combining BP Neural
Network classification and Mahalanobis-Distance-based health scoring for
power generation equipment.*
