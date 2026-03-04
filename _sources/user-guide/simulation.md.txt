# Simulation

## Why this module exists

In production, data doesn't arrive all at once — it streams in continuously.
A detector that works well on a static dataset may behave differently when
processing data in real-time chunks. The Simulation module lets you test
anomaly detection in a realistic streaming scenario, with live-updating
charts that show how the detector responds as new data arrives.

This is valuable for:
- **Validating detector behavior** before deploying to production
- **Tuning thresholds** by watching how scores evolve over time
- **Demonstrating results** to stakeholders with animated visualizations
- **Benchmarking** different configurations (chunk sizes, window sizes, thresholds)

## When to use it

- After you've chosen a detector and want to see how it performs in streaming.
- When preparing for a production deployment and need to validate real-time behavior.
- When presenting anomaly detection results to non-technical audiences.

## How it works

1. Your DataFrame is split into chunks (simulating data arriving over time)
2. Each chunk is preprocessed and fed to an internal Isolation Forest
3. The detector trains on the first full window, then scores each subsequent chunk
4. A live chart updates after each chunk, showing scores and threshold
5. Anomalies (scores above threshold) are highlighted in red

The simulation maintains a rolling history of the last 5,000 data points
and displays the most recent 3 hours of data in the chart.

## Basic usage

```python
import pandas as pd
from sentinel.simulation import StreamingSimulation

sim = StreamingSimulation(
    data=df,                  # time-indexed DataFrame with numeric columns
    chunk_size=50,            # rows per streaming chunk
    stream_interval=0.3,      # seconds between chunks (animation speed)
    window_size=120,          # detector training window size
    threshold=0.15,           # static anomaly threshold
)
```

### Running in Jupyter

```python
sim.run_notebook()
```

This produces a live animated chart inline in the notebook. Each frame
clears the previous output and redraws, creating a smooth animation.
The chart shows:
- **Blue line**: anomaly scores over time (negated, so higher = more anomalous)
- **Orange dashed line**: threshold
- **Red dots**: detected anomalies (scores above threshold)

You can limit the number of steps for quick testing:

```python
sim.run_notebook(max_steps=20)
```

### Running as a standalone script

```python
sim.run()
```

This opens a native matplotlib window with real-time updates. The window
stays open until you press `Ctrl+C`.

## Understanding the parameters

### chunk_size

How many rows are processed at once. This simulates the batch size of
your real-time data pipeline.

| Value | Effect |
|-------|--------|
| Small (10–30) | More frequent updates, smoother animation, slower overall |
| Medium (50–100) | Good balance for most use cases |
| Large (200+) | Fewer updates, faster completion, chunkier animation |

### stream_interval

Seconds between chunks. Controls the animation speed.

| Value | Effect |
|-------|--------|
| 0.1–0.3 | Fast animation — good for demos |
| 0.5–1.0 | Moderate — easier to follow |
| 2.0+ | Slow — useful for detailed observation |

### window_size

Number of data points used to train the internal Isolation Forest. The
detector won't start scoring until it has accumulated this many points.

| Value | Effect |
|-------|--------|
| Small (50–100) | Trains quickly but may be less accurate |
| Medium (120–200) | Good balance |
| Large (500+) | More accurate but takes longer to start scoring |

### threshold

The static anomaly threshold. Scores above this value are flagged as anomalies.

**How to choose:** Run the simulation once with a moderate threshold (0.10–0.15),
observe the score distribution, then adjust. If too many points are flagged,
increase the threshold. If anomalies are missed, decrease it.

## Dynamic thresholds

Instead of a fixed threshold, you can use a percentile-based dynamic threshold
that adapts as more data is processed:

```python
sim = StreamingSimulation(
    data=df,
    chunk_size=50,
    stream_interval=0.3,
    window_size=120,
    threshold=0.15,           # fallback before enough data
    dynamic_threshold=True,
    percentile=95,
)
```

**How it works:**
- The threshold is recalculated at each step as the Nth percentile of all
  historical scores.
- Early in the simulation, when few scores exist, it falls back to the
  static threshold.
- As more data is processed, the dynamic threshold stabilizes.

**When to use dynamic thresholds:**
- When the score distribution changes over time (non-stationary data)
- When you don't know a good static threshold in advance
- When you want the threshold to adapt to the data automatically

| Percentile | Sensitivity |
|-----------|------------|
| 90 | High — flags ~10% of points as anomalies |
| 95 | Balanced — flags ~5% |
| 99 | Low — only extreme anomalies |

## Event overlay

Pass an events DataFrame to overlay incident markers on the chart:

```python
events = pd.DataFrame({
    "start": ["2025-02-11 22:00:00", "2025-02-11 23:59:00"],
    "end": ["2025-02-12 00:00:00", "2025-02-12 00:11:00"],
    "color": ["orange", "red"],
    "label": ["Server restart", "Service outage"],
})

sim = StreamingSimulation(
    data=df,
    chunk_size=50,
    stream_interval=0.3,
    window_size=120,
    threshold=0.15,
    events=events,
)
```

Events appear as colored vertical bands with labels. This lets you visually
correlate detected anomalies with known incidents.

**How to interpret:**
- **Red dots inside event bands** → true positives. The detector caught the incident.
- **Red dots outside event bands** → either unknown incidents or false positives.
- **Event bands without red dots** → the detector missed the incident. Consider
  tuning the threshold or using a different detector.

## Interpreting the animation

As the simulation runs, watch for these patterns:

### Healthy behavior
- Scores stay mostly below the threshold
- Occasional spikes that quickly return to normal
- The threshold line is stable (or slowly adapting if dynamic)

### Anomaly detected
- Scores spike above the threshold (red dots appear)
- If the spike is sustained → ongoing incident
- If the spike is brief → transient event (may or may not be important)

### Detector warming up
- The first few chunks show no scores (the detector is accumulating data)
- Once `window_size` points are collected, scoring begins
- Early scores may be noisy as the detector has limited training data

### Threshold too low
- Many red dots throughout → too many false positives
- Increase `threshold` or `percentile`

### Threshold too high
- No red dots even during obvious events → missing real anomalies
- Decrease `threshold` or `percentile`

## StreamingDataManager

`StreamingDataManager` is the lower-level component that handles chunked
data streaming in a background thread. It is used internally by
`StreamingSimulation` but can also be used independently for custom
streaming pipelines:

```python
from sentinel.simulation.streaming_anomaly_detection import StreamingDataManager

manager = StreamingDataManager(data=df, chunk_size=100, stream_interval=1)
manager.start()

while True:
    chunk = manager.get_next_chunk()
    if chunk is None:
        break
    # Process chunk with your own logic
    print(f"Received {len(chunk)} rows")

manager.stop()
```

## Practical tips

- **Start with `run_notebook(max_steps=10)`** to quickly verify everything works.
- **Use `stream_interval=0.1`** for fast iteration during development.
- **Use `stream_interval=1.0`** for demos and presentations.
- **Compare static vs dynamic thresholds** by running the simulation twice
  with different settings.
- **Save the final `historical_scores`** after simulation for further analysis:
  `np.save("scores.npy", sim.historical_scores)`
