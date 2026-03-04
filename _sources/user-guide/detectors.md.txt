# Detectors

## Why this module exists

Once you've validated that your data contains meaningful signals (via the
Explorer module), the next step is to actually detect anomalies. The Detectors
module provides multiple algorithms with a consistent interface, so you can
compare approaches and pick the one that works best for your data.

## When to use each detector

| Detector | Best for | Data requirements | Speed |
|----------|----------|-------------------|-------|
| `IsolationForestDetector` | General-purpose tabular data | Any numeric features | Fast |
| `RRCFDetector` | Streaming / time series data | Single time series | Medium |
| `AutoencoderDetector` | Complex multivariate patterns | Many numeric features | Slow (training) |
| `LNNDetector` | Complex temporal dynamics | Many numeric features | Slow (training) |

## IsolationForestDetector

The most versatile detector. Based on the Isolation Forest algorithm, which
isolates anomalies by randomly partitioning the feature space. Anomalies
are easier to isolate (fewer partitions needed), so they get lower scores.

### How it works

1. Builds an ensemble of random trees
2. Each tree randomly selects a feature and a split value
3. Anomalies are isolated in fewer splits → shorter path length
4. The anomaly score is the average path length across all trees

### Usage

```python
from sentinel.detectors import IsolationForestDetector

detector = IsolationForestDetector(
    n_estimators=100,       # number of trees (more = more stable)
    contamination=0.05,     # expected fraction of anomalies
    random_state=42,
)
detector.fit(X_train)
```

### Understanding the outputs

```python
# Binary predictions: -1 = anomaly, 1 = normal
predictions = detector.predict(X_test)

# Anomaly scores: lower = more anomalous
scores = detector.decision_function(X_test)

# Get only the anomalous rows
anomalies = detector.get_anomalies(X_test)

# Normalized probability (0 = normal, 1 = anomaly)
proba = detector.predict_proba(X_test)
```

#### Interpreting `decision_function` scores

The scores from `decision_function` are **not bounded** — they can be any
real number. The key insight:

| Score range | Interpretation |
|------------|----------------|
| Strongly negative | Very likely anomaly — far from normal patterns |
| Near zero | Borderline — could go either way |
| Positive | Normal — fits the learned patterns well |

To make scores more intuitive for visualization, negate them:
`df["scores"] = -detector.decision_function(X)` — now higher = more anomalous.

#### Choosing `contamination`

The `contamination` parameter sets the expected fraction of anomalies:

| Value | Effect |
|-------|--------|
| `"auto"` | Let the algorithm decide (conservative) |
| `0.01` (1%) | Very few anomalies expected — strict threshold |
| `0.05` (5%) | Moderate — good starting point |
| `0.10` (10%) | Many anomalies expected — sensitive detection |

If you don't know the anomaly rate, start with `0.05` and adjust based on
the results. Too low → misses real anomalies. Too high → too many false positives.

### When to use it

- First detector to try — works well on most tabular data
- When you need fast training and prediction
- When you want SHAP-based interpretability (tree-based → SHAP compatible)

## RRCFDetector

Robust Random Cut Forest, designed for streaming anomaly detection. Unlike
Isolation Forest, RRCF can update incrementally as new data arrives.

### How it works

1. Builds a forest of random cut trees
2. Each point's anomaly score is its **collusive displacement** — how much
   the tree structure changes when the point is removed
3. Points that significantly alter the tree are anomalous

### Usage

```python
from sentinel.detectors import RRCFDetector

detector = RRCFDetector(
    shingle_size=15,    # temporal context window
    num_trees=100,      # number of trees
    tree_size=500,      # max points per tree
)
scores = detector.fit_predict(series)
anomalies = detector.get_anomalies(threshold=3.0)
```

### Understanding the outputs

```python
# Anomaly scores: higher = more anomalous
scores = detector.fit_predict(series)

# Get anomalies above a threshold
anomalies = detector.get_anomalies(threshold=3.0)

# Normalized scores (0 to 1)
proba = detector.predict_proba(series)
```

#### Interpreting RRCF scores

Unlike Isolation Forest, RRCF scores are **higher for anomalies**:

| Score range | Interpretation |
|------------|----------------|
| Low (< 1.0) | Normal — removing this point barely changes the tree |
| Medium (1.0 – 3.0) | Slightly unusual — worth monitoring |
| High (> 3.0) | Anomalous — this point significantly disrupts the tree structure |

#### Choosing the threshold

There's no universal threshold — it depends on your data. A practical approach:

```python
import numpy as np
scores = detector.fit_predict(series)
# Use the 95th percentile as threshold
threshold = np.percentile(scores, 95)
anomalies = detector.get_anomalies(threshold=threshold)
```

### When to use it

- Time series data with temporal patterns
- Streaming scenarios where data arrives continuously
- When you need incremental updates without full retraining

:::{note}
Requires `pip install -e ".[rrcf]"`. The `rrcf` package depends on
`setuptools<81`.
:::

## AutoencoderDetector

An LSTM autoencoder built on PyTorch. It learns to reconstruct normal
patterns and flags samples with high reconstruction error as anomalies.

### How it works

1. The encoder compresses input features into a latent representation
2. The decoder reconstructs the original features from the latent space
3. Normal data → low reconstruction error (the model learned this pattern)
4. Anomalous data → high reconstruction error (the model hasn't seen this)
5. A threshold is set at `mean(errors) + multiplier * std(errors)`

### Usage

```python
from sentinel.detectors import AutoencoderDetector

detector = AutoencoderDetector(
    n_features=2,              # number of input columns
    seq_len=1,                 # use 1 for multivariate tabular data
    latent_dim=8,              # size of compressed representation
    epochs=30,                 # training iterations
    threshold_multiplier=3.0,  # how many std devs above mean for threshold
)
detector.fit(X_train.values)
```

:::{important}
For multivariate tabular data (not time sequences), always use `seq_len=1`.
The `seq_len` parameter is for sequential data where each sample is a
window of consecutive observations.
:::

### Understanding the outputs

```python
# Binary predictions: 1 = anomaly, 0 = normal
predictions = detector.predict(X_test.values)

# Reconstruction error per sample (higher = more anomalous)
scores = detector.anomaly_score(X_test.values)
```

#### Interpreting reconstruction errors

| Error level | Interpretation |
|------------|----------------|
| Below threshold | Normal — the autoencoder can reconstruct this pattern |
| Slightly above threshold | Borderline — unusual but not extreme |
| Far above threshold | Strong anomaly — the pattern is very different from training data |

#### Choosing `threshold_multiplier`

| Value | Sensitivity |
|-------|------------|
| 2.0 | High sensitivity — catches more anomalies but more false positives |
| 3.0 | Balanced (default) — good starting point |
| 4.0+ | Low sensitivity — only flags extreme anomalies |

### When to use it

- Complex multivariate data where linear methods fail
- When you have enough training data (hundreds+ of normal samples)
- When you want to capture non-linear relationships between features

## LNNDetector

Liquid Neural Network autoencoder using closed-form continuous-time models.
The LNN encoder can capture complex temporal dynamics that standard LSTMs miss.

### How it works

Similar to the AutoencoderDetector, but the encoder uses a Liquid Time-Constant
(LTC) network instead of an LSTM. LTC networks are biologically inspired and
excel at modeling continuous-time dynamics.

### Usage

```python
from sentinel.detectors import LNNDetector

detector = LNNDetector(
    n_features=2,
    seq_len=1,
    latent_dim=8,
    epochs=30,
    threshold_multiplier=3.0,
)
history = detector.fit(X_train.values, verbose=True)

predictions = detector.predict(X_test.values)
scores = detector.anomaly_score(X_test.values)
```

The `history` dict contains `"train"` and optionally `"val"` loss curves,
useful for diagnosing training issues:

```python
import matplotlib.pyplot as plt
plt.plot(history["train"], label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

If the training loss plateaus early, try increasing `epochs` or `latent_dim`.
If it oscillates, reduce `learning_rate`.

### When to use it

- When AutoencoderDetector doesn't capture the patterns well enough
- Data with complex temporal dynamics
- When you want to experiment with cutting-edge architectures

Both deep detectors support model persistence:
```python
detector.save_model("models/my_detector.pt")
detector.load_model("models/my_detector.pt")
```

## Creating a Custom Detector

Subclass `BaseCustomDetector` to implement your own algorithm:

```python
from sentinel.detectors import BaseCustomDetector
import numpy as np

class ZScoreDetector(BaseCustomDetector):
    """Simple Z-score based anomaly detector."""

    def __init__(self, threshold=3.0):
        super().__init__()
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def predict(self, X):
        z = np.abs((X - self.mean_) / self.std_)
        return (z.max(axis=1) > self.threshold).astype(int)

    def anomaly_score(self, X):
        z = np.abs((X - self.mean_) / self.std_)
        return z.max(axis=1)
```

The required methods are `fit()` and `predict()`. The `anomaly_score()`
method is optional but recommended for visualization and threshold tuning.

## Comparing detectors

A practical approach to choosing the right detector:

```python
from sentinel.detectors import IsolationForestDetector
from sentinel.visualization import AnomalyVisualizer

# Try IsolationForest first (fast, interpretable)
iso = IsolationForestDetector(contamination=0.05, random_state=42)
iso.fit(X)
df["iso_scores"] = -iso.decision_function(X)
df["iso_pred"] = iso.predict(X)

# Visualize results
viz = AnomalyVisualizer(df, score_col="iso_scores", anomaly_col="iso_pred")
viz.plot_static(title="Isolation Forest")
viz.plot_score_distribution()

# If results look good → done
# If not → try AutoencoderDetector or LNNDetector
```

## What comes next

After detection:

1. **Visualization** — plot results, understand what was flagged and why
2. **SHAPVisualizer** — explain which features drove each anomaly (tree-based only)
3. **Simulation** — test detection in a streaming scenario
