Sentinel
========

**Signal validation and anomaly detection for enterprise log data.**

Sentinel is a Python library that determines whether unstructured log data
contains meaningful signals before investing resources in complex anomaly
detection pipelines. It provides a modular architecture spanning ingestion,
transformation, exploration, detection, visualization, and simulation.

.. image:: https://github.com/bancolombia/sentinel/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/bancolombia/sentinel/actions/workflows/tests.yml

.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://github.com/bancolombia/sentinel/blob/main/LICENSE

.. image:: https://img.shields.io/badge/python-%3E%3D3.10-blue
   :target: https://www.python.org/

---

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting-started
   user-guide/ingestion
   user-guide/transformer
   user-guide/explorer
   user-guide/detectors
   user-guide/visualization
   user-guide/simulation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/ingestion
   api/transformer
   api/explorer
   api/detectors
   api/visualization
   api/simulation

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
   changelog
