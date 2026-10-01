"""Sentinel: signal validation and anomaly detection for log data.

Sentinel is a Python library for analyzing logs from systems,
applications, and services. It extracts, processes, and analyzes
log data to detect anomalies, patterns, and trends. Its key goal
is to quickly determine whether data contains actionable signals
that could proactively indicate potential problems.

Modules
-------
ingestion
    Transforms raw log files into structured DataFrames.
explorer
    Signal validation and data quality diagnostics.
transformer
    Rolling and string aggregation for time series data.
detectors
    Anomaly detection algorithms (Isolation Forest, RRCF,
    Autoencoder, LNN).
simulation
    Streaming anomaly detection simulation.
visualization
    Anomaly and SHAP visualization utilities.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sentinel")
except PackageNotFoundError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="../..", relative_to=__file__)
    except Exception:  # pragma: no cover - fallback for source trees without tags
        __version__ = "1.0.1"

__author__ = "Arquitectura Innovacion TI"
__license__ = "Apache-2.0"
