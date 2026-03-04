# Getting Started

## Installation

### Prerequisites

- Python ≥ 3.10
- pip (ships with Python)

### Install from source

```bash
git clone https://github.com/bancolombia/sentinel.git
cd sentinel
pip install -e ".[all]"
```

### Install profiles

| Profile | Command | What you get |
|---------|---------|-------------|
| Base | `pip install -e .` | Ingestion, Transformer, Explorer, IsolationForest |
| Dev | `pip install -e ".[dev]"` | + pytest, ipykernel, nbformat |
| Deep | `pip install -e ".[deep]"` | + AutoencoderDetector, LNNDetector (PyTorch) |
| Viz | `pip install -e ".[viz]"` | + AnomalyVisualizer (Plotly), SHAPVisualizer |
| RRCF | `pip install -e ".[rrcf]"` | + RRCFDetector |
| All | `pip install -e ".[all]"` | Everything |

### Verify

```bash
python -c "import sentinel; print(sentinel.__version__)"
pytest -q
```

## Quick Start

```python
import numpy as np
import pandas as pd
from sentinel.explorer import SignalDiagnostics, Thresholds
from sentinel.detectors import IsolationForestDetector
from sentinel.visualization import AnomalyVisualizer

# 1. Generate synthetic data
rng = np.random.RandomState(42)
n = 210
df = pd.DataFrame({
    "cpu": np.concatenate([rng.normal(50, 10, 200), rng.normal(95, 2, 10)]),
    "memory": np.concatenate([rng.normal(2048, 256, 200), rng.normal(7000, 100, 10)]),
}, index=pd.date_range("2025-01-01", periods=n, freq="15min"))

# 2. Signal diagnostics — is there enough signal to detect?
diag = SignalDiagnostics(df, columns=["cpu", "memory"])
report = diag.quality_report(thresholds=Thresholds.relaxed())
print(report)
print(report.interpret())

# 3. Detect anomalies
detector = IsolationForestDetector(contamination=0.05, random_state=42)
detector.fit(df)
df["anomaly"] = detector.predict(df[["cpu", "memory"]])
df["scores"] = -detector.decision_function(df[["cpu", "memory"]])

# 4. Visualize
viz = AnomalyVisualizer(df, score_col="scores", anomaly_col="anomaly")
viz.plot_static(title="Anomaly Detection Results")
```

## Notebooks

Sentinel ships with 8 quickstart notebooks in the `notebooks/` folder:

| # | Notebook | Topic |
|---|----------|-------|
| 01 | Ingestion Quickstart | Log parsing, custom parsers |
| 02 | Transformer Quickstart | Rolling and string aggregation |
| 03 | Explorer Quickstart | Signal diagnostics, drift detection |
| 04 | Detectors Quickstart | IsolationForest, RRCF |
| 05 | Deep Detectors Quickstart | Autoencoder, Liquid Neural Networks |
| 06 | Visualization Quickstart | Static/interactive plots, SHAP |
| 07 | Simulation Quickstart | Streaming anomaly detection |
| 08 | End-to-End Pipeline | Full workflow |

```bash
jupyter notebook notebooks/
```
