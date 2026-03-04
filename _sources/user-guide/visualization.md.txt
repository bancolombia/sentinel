# Visualization

## Why this module exists

Numbers alone don't tell the full story. A detector might flag 50 anomalies,
but are they clustered in time? Do they coincide with known incidents? Which
features drove the detection? The Visualization module answers these questions
through static and interactive charts, score distributions, feature overlays,
and SHAP-based model explanations.

Visualization serves two audiences:
- **Practitioners** who need to validate and tune detection results
- **Stakeholders** who need to understand findings without reading code

## When to use it

- After running any detector, to inspect and validate results.
- When tuning thresholds — the score distribution plot shows separation.
- When explaining results to non-technical audiences.
- When investigating specific anomalies — SHAP plots show which features
  contributed.

## AnomalyVisualizer

`AnomalyVisualizer` renders anomaly scores over time. It requires a
time-indexed DataFrame with score and label columns.

### Setup

```python
from sentinel.visualization import AnomalyVisualizer

viz = AnomalyVisualizer(
    anomaly_df=df,          # time-indexed DataFrame
    score_col="scores",     # column with anomaly scores
    anomaly_col="anomaly",  # column with labels (-1 = anomaly, 1 = normal)
    incidents_df=None,      # optional: DataFrame with incident windows
)
```

If you have incident records, pass them as `incidents_df` with columns
`start_time`, `end_time`, and `Servicio`. They will be overlaid as
shaded regions on the charts.

### Static plot (matplotlib)

```python
viz.plot_static(
    title="Anomaly Detection Results",
    threshold=0.5,
)
```

**What you see:**
- Gray line: anomaly scores over time
- Green dots: normal data points
- Red dots: detected anomalies
- Orange dashed line: threshold (if provided)
- Blue shaded regions: known incidents (if `incidents_df` provided)

**How to interpret:**
- Red dots clustered together → the detector found a sustained anomaly period.
  Check if this aligns with known incidents.
- Scattered red dots → isolated spikes. Could be real anomalies or noise.
  Look at the score values — are they far above the threshold?
- Red dots during blue shaded regions → true positives (the detector caught
  known incidents). This validates the detector.
- Red dots outside blue regions → either unknown incidents or false positives.
  Investigate these manually.

You can zoom into a specific date range:

```python
viz.plot_static(zoom=True, zoom_date=["2025-01-10", "2025-01-12"])
```

### Interactive plot (Plotly)

```python
viz.plot_dynamic(threshold=0.5)
```

Same information as the static plot, but interactive — you can zoom, pan,
hover over points to see exact values, and toggle traces on/off. Best for
exploratory analysis in Jupyter notebooks.

### Score distribution histogram

```python
viz.plot_score_distribution(threshold=0.5)
```

**What you see:**
- Blue histogram: score distribution for normal data points
- Red histogram: score distribution for anomalies
- Orange dashed line: threshold (if provided)

**How to interpret:**

This is the most important plot for **threshold tuning**:

- **Good separation** (little overlap between blue and red) → the detector
  can reliably distinguish anomalies. Your threshold should sit in the gap.
- **Significant overlap** → the detector struggles to separate normal from
  anomalous. Consider: different features, different detector, or the data
  may not have clear anomalies.
- **Threshold too far left** → you're catching anomalies but also many
  false positives (normal points above threshold).
- **Threshold too far right** → you're missing real anomalies (red bars
  below threshold).

The ideal threshold sits where:
- Most normal scores are below it
- Most anomaly scores are above it
- The false positive rate is acceptable for your use case

### Feature overlay

```python
viz.plot_features(feature_columns=["cpu", "memory"])
```

**What you see:**
- One subplot per feature, sharing the time axis
- Blue line: raw feature values over time
- Red dots: values at time points classified as anomalies

**How to interpret:**
- Red dots at feature peaks/valleys → the anomaly was driven by extreme
  values in this feature.
- Red dots at normal-looking values → the anomaly was driven by a
  *combination* of features, not this one alone. Check other subplots.
- Consistent red dots across all features → a systemic event affected
  everything simultaneously.

This plot answers the question: "What was happening in the data when the
anomaly was detected?"

## SHAPVisualizer

`SHAPVisualizer` explains predictions of tree-based detectors using SHAP
(SHapley Additive exPlanations) values. It answers: **why did the detector
flag this specific data point?**

### Setup

```python
from sentinel.visualization import SHAPVisualizer

shap_viz = SHAPVisualizer(detector)  # pass a fitted IsolationForestDetector
```

SHAP values are cached after the first computation, so subsequent plots
on the same data are fast.

### Summary plot (beeswarm)

```python
shap_viz.plot_summary(X)
```

**What you see:**
- One row per feature, sorted by importance (most important at top)
- Each dot is one data point
- X axis: SHAP value (impact on the anomaly score)
- Color: feature value (red = high, blue = low)

**How to interpret:**
- Features at the top have the most influence on detection.
- If red dots (high values) push SHAP right (positive) → high values of
  this feature make the point more anomalous.
- If blue dots (low values) push SHAP right → low values are anomalous.
- A wide spread of dots → the feature has variable impact across samples.
- A narrow cluster → the feature has consistent, predictable impact.

**Example insight:** "CPU has the highest importance. High CPU values
(red dots) push the score toward anomaly (right), confirming that CPU
spikes are the primary anomaly driver."

### Bar chart (global importance)

```python
shap_viz.plot_bar(X)
```

**What you see:**
- Horizontal bars showing mean |SHAP value| per feature
- Sorted by importance

**How to interpret:**
- Longer bar = more important feature for detection overall.
- This is a simpler view than the summary plot — use it when you just
  need the importance ranking without distribution details.

### Waterfall chart (single sample)

```python
# Explain a specific anomaly
anomaly_indices = df[df["anomaly"] == -1].index
shap_viz.plot_waterfall(X, sample_index=0)
```

**What you see:**
- Starting from the base value (expected output)
- Each feature adds or subtracts from the prediction
- Red bars push toward anomaly, blue bars push toward normal
- Final value at the top

**How to interpret:**
- "The base prediction was -0.05. CPU added +0.12 (pushing toward anomaly),
  memory added +0.08, but error_count subtracted -0.02. Final score: 0.13."
- This tells you exactly which features made this specific point anomalous.

### Force plot (single sample)

```python
shap_viz.plot_force(X, anomaly_index=0)
```

Similar to the waterfall but displayed horizontally. Features pushing
toward anomaly are in red, features pushing toward normal are in blue.
The width of each segment shows the magnitude of the contribution.

### Dependence plot

```python
shap_viz.plot_dependence(X, feature="cpu")
```

**What you see:**
- Scatter plot: feature value (X axis) vs SHAP value (Y axis)
- Color: interaction feature (automatically selected or specified)

**How to interpret:**
- **Linear trend** → the feature has a consistent, proportional effect.
- **Non-linear curve** → the feature's impact changes at certain values.
  For example, "CPU only matters when it exceeds 80%."
- **Color patterns** → interactions between features. "CPU matters more
  when memory is also high (red dots at top right)."

```python
# Specify the interaction feature explicitly
shap_viz.plot_dependence(X, feature="cpu", interaction_feature="memory")
```

## Visualization workflow

A recommended sequence for analyzing detection results:

```
1. plot_static()              → Overview: where are the anomalies?
2. plot_score_distribution()  → Threshold tuning: is the separation good?
3. plot_features()            → Context: what happened in the data?
4. plot_summary()             → Global: which features matter most?
5. plot_waterfall()           → Local: why was THIS point flagged?
6. plot_dependence()          → Deep dive: how does a feature affect scores?
```

## Tips for effective visualization

- **Always negate Isolation Forest scores** before plotting:
  `df["scores"] = -detector.decision_function(X)` — this makes higher = more anomalous.
- **Start with `plot_score_distribution`** to understand the score landscape
  before choosing a threshold.
- **Use `plot_features` to validate** — if anomaly markers don't align with
  visible patterns in the features, the detector may need tuning.
- **SHAP plots require tree-based detectors** — they won't work with
  AutoencoderDetector or LNNDetector.
