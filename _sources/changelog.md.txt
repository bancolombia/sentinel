# Changelog

## 0.1.0 (2026-03-04)

Initial release.

- Ingestion module with parsers for WAS, HSM, HDC, IBMMQ, and ZTNA log formats
- Transformer module with RollingAggregator and StringAggregator
- Explorer module with SignalDiagnostics, Thresholds, QualityReport, drift detection
- Detectors: IsolationForestDetector, RRCFDetector, AutoencoderDetector, LNNDetector
- Visualization: AnomalyVisualizer (matplotlib + Plotly), SHAPVisualizer
- Simulation: StreamingSimulation with Jupyter and standalone support
- 8 quickstart notebooks
- JOSS paper draft
- CI with GitHub Actions (pytest + JOSS draft PDF)
