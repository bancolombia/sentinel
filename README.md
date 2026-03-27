<p align="center">
  <img src="docs/images/sentinel icon with text.png" alt="Sentinel Logo" width="200"/>
</p>

# Sentinel

[![CI](https://github.com/bancolombia/sentinel/actions/workflows/tests.yml/badge.svg)](https://github.com/bancolombia/sentinel/actions/workflows/tests.yml)
[![Paper PDF](https://github.com/bancolombia/sentinel/actions/workflows/draft-pdf.yml/badge.svg)](https://github.com/bancolombia/sentinel/actions/workflows/draft-pdf.yml)
[![DOI](https://zenodo.org/badge/1162599856.svg)](https://doi.org/10.5281/zenodo.19259903)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)

Signal validation and anomaly detection for enterprise log data.

Sentinel is a Python library that determines whether unstructured log data
contains meaningful signals before investing resources in complex anomaly
detection pipelines. It provides a modular architecture spanning **ingestion,
transformation, exploration, detection, visualization, and simulation**.

---

## Installation

```bash
# Base install (includes IsolationForest, explorer, transformer)
pip install .

# Development (pytest, ipykernel, nbformat)
pip install -e ".[dev]"

# Deep learning detectors (AutoencoderDetector, LNNDetector)
pip install ".[deep]"

# Visualization (plotly, SHAP, matplotlib)
pip install ".[viz]"

# Robust Random Cut Forest detector
pip install ".[rrcf]"

# Everything
pip install -e ".[all]"
```

---

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
viz.plot_score_distribution(threshold=df.loc[df["anomaly"] == 1, "scores"].mean())
```

---

## Modules

### Ingestion

Transforms raw log files into structured DataFrames. Use a built-in parser
directly or dispatch via `LogIngestor`.

| Parser | Log Format |
|--------|-----------|
| `WASParser` | WebSphere Application Server |
| `HSMParser` | Hardware Security Module |
| `HDCParser` | High-Density Computing |
| `IBMMQParser` | IBM Message Queue |
| `ZTNAParser` | Cloudflare Zero Trust Network Access |

```python
from sentinel.ingestion import LogIngestor

# Quick dispatch
df = LogIngestor.ingest("path/to/logfile.log", log_type="WAS")

# Or use a parser directly
from sentinel.ingestion import WASParser
parser = WASParser("path/to/logfile.log")
df = parser.parse()
```

Custom parsers can be created by subclassing `BaseLogParser`:

```python
from sentinel.ingestion import BaseLogParser
import pandas as pd

class MyParser(BaseLogParser):
    def parse(self):
        records = []
        with open(self.file_path) as f:
            for line in f:
                # your parsing logic here
                records.append({"timestamp": ..., "value": ...})
        return pd.DataFrame(records)

parser = MyParser("path/to/custom.log")
df = parser.parse()
```

### Transformer

Rolling and string-based aggregation for time series feature engineering.

```python
from sentinel.transformer import RollingAggregator, StringAggregator

# Rolling statistics over numeric columns
agg = RollingAggregator(window_size=12, aggregation_functions="mean")
rolled = agg.fit_transform(numeric_df)

# Categorical aggregation with time windows
str_agg = StringAggregator(df, timestamp_column="timestamp")
counts = str_agg.create_time_aggregation(
    time_window="1h",
    column_metrics={"status": ["count", "nunique"]},
    category_count_columns={"status": ["OK", "ERROR"]},
)
```

### Explorer

Signal validation and data quality diagnostics. Checks whether your data
has enough variance, sufficient entries, and detectable outliers before
you invest compute in detection.

```python
from sentinel.explorer import SignalDiagnostics, Thresholds, detect_drift

# Quality report with interpretable results
diag = SignalDiagnostics(df, columns=["cpu", "memory"])
report = diag.quality_report(thresholds=Thresholds.relaxed())
print(f"Score: {report.score:.0%}")
print(report.interpret())

# Distribution drift detection (Kolmogorov-Smirnov)
drift_results = detect_drift(df, column="cpu", window=200)
for r in drift_results:
    print(f"Window [{r['start_idx']}:{r['end_idx']}] — drifted: {r['drifted']}")
```

### Detectors

| Detector | Algorithm | Install group |
|----------|-----------|--------------|
| `IsolationForestDetector` | Isolation Forest | base |
| `RRCFDetector` | Robust Random Cut Forest | `rrcf` |
| `AutoencoderDetector` | LSTM Autoencoder | `deep` |
| `LNNDetector` | Liquid Neural Network | `deep` |
| `BaseCustomDetector` | Abstract base for custom detectors | base |

```python
from sentinel.detectors import IsolationForestDetector

detector = IsolationForestDetector(contamination=0.05, random_state=42)
detector.fit(X_train)
predictions = detector.predict(X_test)       # -1 = anomaly, 1 = normal
scores = detector.decision_function(X_test)  # lower = more anomalous
```

### Visualization

Static (matplotlib) and interactive (Plotly) anomaly plots, score
distribution histograms, feature overlays, and SHAP-based model
interpretability.

```python
from sentinel.visualization import AnomalyVisualizer, SHAPVisualizer

# AnomalyVisualizer — score timeline, distribution, feature overlay
viz = AnomalyVisualizer(anomaly_df, score_col="scores", anomaly_col="anomaly")
viz.plot_static(threshold=0.5)              # matplotlib scatter with threshold line
viz.plot_dynamic(threshold=0.5)             # interactive Plotly chart
viz.plot_score_distribution(threshold=0.5)  # histogram of normal vs anomaly scores
viz.plot_features()                         # feature time series with anomaly markers

# SHAPVisualizer — model interpretability (tree-based detectors)
shap_viz = SHAPVisualizer(detector)
shap_viz.plot_summary(X)       # beeswarm: global feature importance
shap_viz.plot_bar(X)           # bar chart: mean |SHAP| per feature
shap_viz.plot_waterfall(X, 0)  # waterfall: single sample explanation
shap_viz.plot_force(X, 0)      # force plot: single sample
shap_viz.plot_dependence(X, feature="cpu")  # feature dependence scatter
```

### Simulation

`StreamingSimulation` simulates real-time data streaming with live anomaly
detection and visualization. Works in both Jupyter notebooks and standalone
scripts.

```python
from sentinel.simulation import StreamingSimulation

sim = StreamingSimulation(
    data=df,
    chunk_size=50,
    stream_interval=0.3,
    window_size=120,
    threshold=0.15,
    dynamic_threshold=True,
    percentile=95,
    events=events_df,  # optional incident overlay
)

# In Jupyter — inline animated chart
sim.run_notebook()

# In a terminal script — native matplotlib window
# sim.run()
```

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Ingestion Quickstart](notebooks/01_ingestion_quickstart.ipynb) | Built-in parsers (WAS, HSM), LogIngestor dispatch, custom parser with `BaseLogParser` |
| 02 | [Transformer Quickstart](notebooks/02_transformer_quickstart.ipynb) | `RollingAggregator` and `StringAggregator` with custom metrics |
| 03 | [Explorer Quickstart](notebooks/03_explorer_quickstart.ipynb) | `SignalDiagnostics`, `QualityReport` with `interpret()`, drift detection |
| 04 | [Detectors Quickstart](notebooks/04_detectors_quickstart.ipynb) | `IsolationForestDetector` and `RRCFDetector` with `AnomalyVisualizer` |
| 05 | [Deep Detectors Quickstart](notebooks/05_deep_detectors_quickstart.ipynb) | `AutoencoderDetector` and `LNNDetector` with visualization |
| 06 | [Visualization Quickstart](notebooks/06_visualization_quickstart.ipynb) | Full `AnomalyVisualizer` suite + all `SHAPVisualizer` methods |
| 07 | [Simulation Quickstart](notebooks/07_simulation_quickstart.ipynb) | `StreamingSimulation` with live animated charts (static and dynamic thresholds) |
| 08 | [End-to-End Pipeline](notebooks/08_end_to_end_pipeline.ipynb) | Complete pipeline: ingestion → transformation → exploration → detection → visualization → SHAP |

---

## Project Structure

```
sentinel/
├── src/sentinel/
│   ├── ingestion/       # Log parsers (WAS, HSM, HDC, IBMMQ, ZTNA, base)
│   ├── transformer/     # RollingAggregator, StringAggregator
│   ├── explorer/        # SignalDiagnostics, Thresholds, drift detection
│   ├── detectors/       # IsolationForest, RRCF, Autoencoder, LNN, custom base
│   ├── visualization/   # AnomalyVisualizer, SHAPVisualizer
│   └── simulation/      # StreamingSimulation, StreamingDataManager
├── tests/               # pytest test suite
├── notebooks/           # Quickstart notebooks (01–08)
├── examples/            # Standalone example scripts
├── paper/               # JOSS paper source
└── pyproject.toml       # Dependencies and build config
```

---

## Paper

The JOSS paper is in `paper/`. A draft PDF is built automatically on each push via GitHub Actions.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Citation

```bibtex
@software{sentinel2026,
  title   = {Sentinel: Signal Validation and Anomaly Detection for Enterprise Log Data},
  author  = {Vergara Álvarez, José Manuel and Laverde Manotas, Nicolás and Aguilar Calle, Juan Pablo and Niño Castillo, Jeisson Vicente and Muñoz Pertuz, Julián David and Monsalve Muñoz, Daniel and Osorio Agudelo, Sebastián},
  year    = {2026},
  url     = {https://github.com/bancolombia/sentinel}
}
```
