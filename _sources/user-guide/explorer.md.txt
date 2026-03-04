# Explorer

## Why this module exists

Not all data is worth analyzing. Enterprise logs can contain columns that are
nearly constant, mostly null, or completely uncorrelated with known incidents.
Running anomaly detection on such data wastes compute and produces meaningless
results. The Explorer module answers a critical question before you invest in
detection: **does this data contain enough signal to detect anomalies?**

This "fail-fast" approach can save hours of wasted effort. If the Explorer
says the data lacks signal, you know to collect better data or engineer
better features before proceeding.

## When to use it

- After ingestion and transformation, before running detectors.
- When you're unsure whether your features are informative.
- When you want a quick, automated data quality assessment.
- When you need to justify to stakeholders why certain data sources
  are (or aren't) suitable for anomaly detection.

## SignalDiagnostics

`SignalDiagnostics` is the main entry point. It takes a DataFrame and a list
of numeric columns to analyze.

```python
from sentinel.explorer import SignalDiagnostics, Thresholds

diag = SignalDiagnostics(
    df,
    columns=["cpu", "memory", "error_count"],
    label_column="label",           # optional: binary 0/1 column
    timestamp_column="timestamp",   # optional
)
```

### Quality Report

The quality report runs a battery of checks on each column and returns a
structured result:

```python
report = diag.quality_report(thresholds=Thresholds.relaxed())
print(report)
# QualityReport(PASSED, 12 checks, 0 failed)

print(f"Score: {report.score:.0%}")
# Score: 100%

print(report.interpret())
```

#### What the checks mean

| Check | What it measures | Why it matters |
|-------|-----------------|----------------|
| `min_entries` | Number of non-null data points | Too few rows → unreliable statistics, overfitting risk |
| `min_non_null_pct` | Percentage of non-null values | High null rates → the column may be unreliable or sparse |
| `min_variance` | Statistical variance of the column | Near-zero variance → the column is nearly constant, nothing to detect |
| `anomaly_pct` | Percentage of IQR outliers | No outliers → the distribution is too uniform for anomaly detection |
| `correlation` | Point-biserial correlation with label | Low correlation → the feature doesn't relate to known anomalies |

#### Interpreting the report

The `interpret()` method returns a human-readable explanation:

**When all checks pass:**
```
SIGNAL DETECTED — All 12 checks passed (score: 100%).
The data has sufficient volume, variance, and anomaly presence
to proceed with anomaly detection.
```

**When checks fail:**
```
INSUFFICIENT SIGNAL — 3/12 checks failed (score: 75%).

  [min_variance] Failed for: error_count
    → Near-constant signal. The column has very low variance...

  [anomaly_pct] Failed for: cpu, memory
    → Too few IQR outliers detected...

Recommendation: Review whether low-variance columns carry useful
information. Drop or transform them before detection.
```

#### What to do when checks fail

| Failed check | Action |
|-------------|--------|
| `min_entries` | Collect more data, or use `Thresholds.relaxed()` for exploration |
| `min_non_null_pct` | Investigate the data source, impute missing values, or drop the column |
| `min_variance` | The column is nearly constant — drop it or engineer a derived feature |
| `anomaly_pct` | The data may be too uniform — try different features or time windows |
| `correlation` | The feature isn't predictive — add more informative features or review labels |

### Thresholds

Thresholds control how strict the quality checks are. Three presets are
provided:

| Preset | `min_entries` | `min_non_null_pct` | `min_variance` | `anomaly_pct` | Best for |
|--------|-------------|-------------------|----------------|--------------|----------|
| `default()` | 1,000 | 95% | 0.01 | 5% | Production-grade data |
| `strict()` | 25,000 | 99% | 500 | 5% | Large, high-quality datasets |
| `relaxed()` | 100 | 80% | 0.001 | 1% | Exploratory analysis, small samples |

You can also create custom thresholds:

```python
custom = Thresholds(
    min_entries=500,
    min_non_null_pct=90.0,
    min_variance=0.01,
    correlation_threshold=0.2,
    anomaly_pct_threshold=2.0,
)
report = diag.quality_report(thresholds=custom)
```

### Statistical summary

Get per-column statistics for manual inspection:

```python
summary = diag.summary()
for col, stats in summary.items():
    print(f"\n{col}:")
    print(f"  Count:          {stats['count']}")
    print(f"  Null %:         {stats['null_pct']}%")
    print(f"  Mean:           {stats['mean']:.2f}")
    print(f"  Std:            {stats['std']:.2f}")
    print(f"  Variance:       {stats['variance']:.2f}")
    print(f"  IQR anomalies:  {stats['iqr_anomaly_count']} ({stats['iqr_anomaly_pct']}%)")
```

#### How to read the summary

- **High `null_pct`** (>20%) → the column has significant missing data.
  Consider whether this is structural (expected) or a data quality issue.
- **Low `variance`** (<0.01) → the column barely changes. Detectors won't
  find meaningful patterns here.
- **High `iqr_anomaly_pct`** (>10%) → many outliers. Could mean the data
  is genuinely anomalous, or the distribution is heavy-tailed.
- **`iqr_anomaly_pct` = 0%** → no outliers at all. The data may be too
  clean or the column may not be informative.

### Anomaly distribution

Quick IQR-based outlier analysis per column:

```python
dist = diag.anomaly_distribution()
# {'cpu': {'method': 'iqr', 'anomaly_count': 15, 'anomaly_pct': 7.14}}
```

This tells you how many data points fall outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`.
A healthy dataset for anomaly detection typically has 1–10% IQR outliers.

### Correlation report (requires label column)

If you have a binary label column (0 = normal, 1 = anomaly), the correlation
report measures how well each feature relates to the labels:

```python
diag = SignalDiagnostics(df, columns=["cpu", "memory"], label_column="label")
corr = diag.correlation_report()
# {'cpu': {'point_biserial': 0.72, 'p_value': 1.2e-15}}
```

### Interpreting correlation values

| Correlation | Interpretation |
|------------|----------------|
| > 0.5 | Strong signal — this feature is highly predictive |
| 0.3 – 0.5 | Moderate signal — useful in combination with other features |
| 0.1 – 0.3 | Weak signal — may contribute marginally |
| < 0.1 | No signal — this feature is not predictive of anomalies |

The p-value tells you whether the correlation is statistically significant.
Values < 0.05 are generally considered significant.

### Predictive power (requires label column)

Evaluates each feature individually as a predictor using logistic regression:

```python
power = diag.predictive_power()
# {'cpu': {'recall': 0.85}, 'memory': {'recall': 0.60}}
```

**Recall** measures what fraction of actual anomalies the single-feature
model correctly identifies:

| Recall | Interpretation |
|--------|----------------|
| > 0.7 | Strong predictor on its own |
| 0.4 – 0.7 | Moderate — useful in a multi-feature model |
| < 0.4 | Weak — unlikely to help detect anomalies alone |

### Quick report (all-in-one)

Run all diagnostics at once:

```python
full = diag.quick_report(thresholds=Thresholds.relaxed())
print(full["interpretation"])
# Access individual sections:
# full["summary"], full["quality_report"], full["correlation"],
# full["anomaly_distribution"], full["predictive_power"]
```

## Drift Detection

`detect_drift` uses the Kolmogorov-Smirnov (KS) test to detect distribution
shifts in a sliding window. This is useful for identifying when the
statistical properties of your data change over time — a common precursor
to anomalies.

```python
from sentinel.explorer import detect_drift

results = detect_drift(df, column="cpu", window=200)
for r in results:
    print(f"Window [{r['start_idx']}:{r['end_idx']}]")
    print(f"  KS statistic: {r['statistic']}")
    print(f"  p-value:      {r['p_value']}")
    print(f"  Drifted:      {r['drifted']}")
```

### Interpreting drift results

| Field | Meaning |
|-------|---------|
| `statistic` | KS test statistic (0–1). Higher = more different distributions |
| `p_value` | Probability that the two windows come from the same distribution |
| `drifted` | `True` if p_value < 0.05 (statistically significant drift) |

**Why drift matters for anomaly detection:**
- If the data distribution shifts, a model trained on old data may produce
  false positives or miss real anomalies.
- Drift detection helps you decide when to retrain your detector.
- Sudden drift often coincides with system changes (deployments, config
  changes, failures).

## IQR Anomaly Detection (standalone)

A simple, fast outlier detection function:

```python
from sentinel.explorer import detect_anomalies

outliers = detect_anomalies(df, column="cpu")
print(f"Found {len(outliers)} outliers out of {len(df)} rows")
```

This returns the subset of rows where the value falls outside
`[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`. It's useful for quick sanity checks
but not a substitute for the full detector algorithms.

## Loading and Labeling Data

`load_and_label` is a convenience function for loading CSV files and
optionally labeling rows based on an events file:

```python
from sentinel.explorer import load_and_label

df, columns = load_and_label(
    "data.csv",
    columns=["cpu", "memory"],
    events_csv_path="events.csv",
    timestamp_column="timestamp",
)
# df now has a "label" column: 1 = within an event window, 0 = normal
```

This is particularly useful when you have incident records and want to
create labeled data for correlation analysis and predictive power evaluation.

## Decision framework

```
┌─────────────────────────────────┐
│  Run quality_report()           │
└──────────┬──────────────────────┘
           │
     ┌─────▼─────┐
     │  Passed?   │
     └──┬─────┬───┘
        │     │
      Yes     No
        │     │
        ▼     ▼
   Proceed   Check interpret()
   with      ─── fix data issues
   Detectors ─── try relaxed thresholds
             ─── collect more data
```
