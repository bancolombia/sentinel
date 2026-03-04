# Transformer

## Why this module exists

Raw log data is typically a sequence of individual events — one row per log
line. Anomaly detectors, however, work best on **aggregated features** that
capture patterns over time windows (e.g., "average CPU over the last hour"
or "number of ERROR messages in the last 5 minutes"). The Transformer module
bridges this gap by converting raw event data into time-windowed numeric
features suitable for detection algorithms.

## When to use it

- After ingestion, when you have a time-indexed DataFrame of raw events.
- When you need rolling statistics (mean, std, min, max) over sliding windows.
- When you need to count categorical values (e.g., how many "ERROR" vs "OK"
  messages per time window).
- When preparing features for any of the Detectors.

## RollingAggregator

`RollingAggregator` computes rolling, expanding, or exponentially weighted
moving window aggregations on numeric columns. It follows the scikit-learn
transformer API (`fit` / `transform` / `fit_transform`).

### Basic usage

```python
import numpy as np
import pandas as pd
from sentinel.transformer import RollingAggregator

# Sample data
df = pd.DataFrame({
    "cpu": np.random.normal(50, 10, 500),
    "memory": np.random.normal(2048, 256, 500),
}, index=pd.date_range("2025-01-01", periods=500, freq="15min"))

agg = RollingAggregator(
    window_size=12,
    aggregation_functions=["mean", "std"],
    columns=["cpu", "memory"],
)
result = agg.fit_transform(df)
print(result.columns.tolist())
# ['cpu', 'memory', 'cpu_rolling_mean', 'cpu_rolling_std',
#  'memory_rolling_mean', 'memory_rolling_std']
```

### Understanding the output

The transformer **preserves the original columns** and adds new ones with
the naming pattern `{column}{suffix}_{function}`. By default the suffix
is `_rolling`.

| Output column | Meaning |
|--------------|---------|
| `cpu_rolling_mean` | Average CPU over the last 12 data points |
| `cpu_rolling_std` | Standard deviation of CPU over the last 12 points |

The first `window_size - 1` rows will contain `NaN` values because there
aren't enough preceding points to fill the window. This is expected — you
can either drop these rows or use `min_periods` to allow partial windows:

```python
agg = RollingAggregator(window_size=12, min_periods=1, ...)
```

### Window types

| Type | Behavior | Best for |
|------|----------|----------|
| `"fixed"` (default) | Standard sliding window of fixed size | Most use cases — stable, predictable features |
| `"expanding"` | Window grows from the start (cumulative) | Baseline statistics that incorporate all history |
| `"ewm"` | Exponentially weighted — recent points matter more | Reactive features that adapt quickly to changes |

```python
# Expanding: cumulative mean from the start
agg = RollingAggregator(window_size=12, window_type="expanding",
                        aggregation_functions="mean")

# EWM: exponentially weighted mean with span=12
agg = RollingAggregator(window_size=12, window_type="ewm",
                        aggregation_functions="mean")
```

### Choosing the right window size

The window size determines how much history each feature captures:

| Window size | Effect | Use case |
|-------------|--------|----------|
| Small (3–10) | Sensitive to short-term spikes | Detecting sudden bursts or drops |
| Medium (12–60) | Balances noise and signal | General-purpose anomaly detection |
| Large (100+) | Smooth, captures long-term trends | Detecting slow drifts or seasonal shifts |

A good rule of thumb: if your data is sampled every 15 minutes and you want
hourly features, use `window_size=4`. For daily features, use `window_size=96`.

### Custom aggregation functions

You can mix string names with callable functions:

```python
import numpy as np

def iqr(x):
    return np.percentile(x, 75) - np.percentile(x, 25)

agg = RollingAggregator(
    window_size=12,
    aggregation_functions=["mean", "std", np.median, iqr],
    columns=["cpu"],
)
```

### Interpreting rolling features for anomaly detection

Rolling features are powerful because they capture **context**:

- **High `_rolling_std`** → the signal is volatile in this window. If the
  detector sees a window with unusually high std, it may flag it.
- **Divergence between `_rolling_mean` and raw value** → the current value
  deviates from the recent trend. Large divergence = potential anomaly.
- **Sudden drop in `_rolling_mean`** → the system behavior changed. Could
  indicate a failure or a configuration change.

## StringAggregator

`StringAggregator` performs time-window aggregation on DataFrames with
categorical or string columns. It's designed for log data where you need
to count events, track unique values, or measure event frequency.

### Basic usage

```python
from sentinel.transformer import StringAggregator

str_agg = StringAggregator(df, timestamp_column="timestamp")
result = str_agg.create_time_aggregation(
    time_window="1h",
    column_metrics={"status": ["count", "nunique"]},
    category_count_columns={"status": ["OK", "ERROR", "TIMEOUT"]},
)
```

### Understanding the output

The output is a DataFrame indexed by time window boundaries:

| Output column | Meaning |
|--------------|---------|
| `status_count` | Total number of log entries in this time window |
| `status_nunique` | Number of distinct status values seen |
| `status_OK_count` | How many entries had status = "OK" |
| `status_ERROR_count` | How many entries had status = "ERROR" |
| `avg_time_between_events_seconds` | Average gap between consecutive events |
| `min_time_between_events_seconds` | Shortest gap (burst detection) |
| `max_time_between_events_seconds` | Longest gap (silence detection) |

### Interpreting the results

These aggregated features tell you about the **rhythm** of your system:

- **`status_ERROR_count` spikes** → something went wrong in that time window.
  If this correlates with anomalies detected later, you've found a useful signal.
- **`avg_time_between_events_seconds` increases** → the system is slowing down
  or producing fewer events. Could indicate degraded performance.
- **`min_time_between_events_seconds` near zero** → burst of events in rapid
  succession. Often seen during cascading failures.
- **`status_nunique` drops to 1** → the system is stuck in a single state
  (e.g., only producing ERROR messages). Strong anomaly indicator.

### Available metrics

| Metric | Type | Description |
|--------|------|-------------|
| `"count"` | string | Number of non-null values |
| `"nunique"` | string | Number of unique values |
| `"mode"` | string | Most frequent value |
| callable | function | Any custom function applied to the group |

### Custom global metrics

```python
def error_ratio(group):
    total = len(group)
    errors = (group["status"] == "ERROR").sum()
    return errors / total if total > 0 else 0

result = str_agg.create_time_aggregation(
    time_window="1h",
    column_metrics={"status": ["count"]},
    custom_metrics={"error_ratio": error_ratio},
)
```

### Choosing the time window

| Window | Resolution | Use case |
|--------|-----------|----------|
| `"1min"` | Very fine | High-frequency systems, real-time monitoring |
| `"5min"` – `"15min"` | Fine | Standard operational monitoring |
| `"1h"` | Medium | Hourly trend analysis |
| `"1D"` | Coarse | Daily summaries, long-term patterns |

Smaller windows give more data points but noisier features. Larger windows
are smoother but may hide short-lived anomalies.

## What comes next

After transformation, your data is ready for:

1. **Explorer** — validate signal quality before investing in detection
2. **Detectors** — run anomaly detection on the aggregated features
