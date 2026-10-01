"""Visualization module for anomaly detection results and SHAP analysis.

Provides ``AnomalyVisualizer`` for plotting anomaly scores over time
and ``SHAPVisualizer`` for model interpretability via SHAP values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap


class AnomalyVisualizer:
    """Visualize anomaly detection results with static and interactive plots.

    Parameters
    ----------
    anomaly_df : pd.DataFrame
        Time-indexed DataFrame with anomaly scores and labels.
    incidents_df : pd.DataFrame, optional
        DataFrame with ``start_time``, ``end_time``, and ``Servicio`` columns.
    score_col : str, default='scores'
        Column name for anomaly scores.
    anomaly_col : str, default='anomaly'
        Column name for anomaly labels (``-1`` = anomaly).
    """

    def __init__(self, anomaly_df, incidents_df=None, score_col='scores', anomaly_col='anomaly'):
        self.anomaly_df = anomaly_df
        self.incidents_df = incidents_df
        self.score_col = score_col
        self.anomaly_col = anomaly_col

    def plot_static(self, zoom=False, zoom_date=None, threshold=None,
                    colors=None, title=None, xlabel=None, ylabel=None, legend_labels=None):
        """Plot a static scatter chart using matplotlib.

        Parameters
        ----------
        zoom : bool, default=False
            Whether to zoom into a specific date range.
        zoom_date : list of str, optional
            ``[start_date, end_date]`` for zooming.
        threshold : float, optional
            If provided, draws a horizontal threshold line.
        colors : dict, optional
            Colors for ``'normal'``, ``'anomaly'``, and ``'incident'``.
        title : str, optional
            Chart title.
        xlabel : str, optional
            X-axis label.
        ylabel : str, optional
            Y-axis label.
        legend_labels : list of str, optional
            Custom legend labels.
        """
        if colors is None:
            colors = {'normal': 'green', 'anomaly': 'red', 'incident': 'blue'}
        if legend_labels is None:
            legend_labels = ['Normal', 'Anomaly', 'Incident']

        plt.figure(figsize=(12, 6))
        plt.plot(self.anomaly_df.index, self.anomaly_df[self.score_col], 'gray', alpha=0.5)
        plt.scatter(
            self.anomaly_df.index, self.anomaly_df[self.score_col],
            c=[colors['anomaly'] if x == -1 else colors['normal'] for x in self.anomaly_df[self.anomaly_col]],
            alpha=0.5
        )

        if threshold is not None:
            plt.axhline(y=threshold, color='orange', linestyle='--', linewidth=1.2, label=f'Threshold ({threshold:.3f})')

        if self.incidents_df is not None:
            for _, row in self.incidents_df.iterrows():
                plt.axvspan(row['start_time'], row['end_time'], color=colors['incident'], alpha=0.1)

        plt.xlabel(xlabel or 'Time')
        plt.ylabel(ylabel or 'Anomaly Score')
        plt.title(title or 'Detected Anomalies vs Incidents')
        plt.xticks(rotation=45)
        plt.legend(legend_labels, loc='upper left')

        if zoom and zoom_date:
            plt.xlim(pd.Timestamp(zoom_date[0]), pd.Timestamp(zoom_date[1]))

        plt.tight_layout()
        plt.show()

    def plot_dynamic(self, threshold=None, colors=None, title=None, xaxis_title=None, yaxis_title=None):
        """Plot an interactive chart using Plotly.

        Parameters
        ----------
        threshold : float, optional
            If provided, draws a horizontal threshold line.
        colors : dict, optional
            Colors for ``'normal'``, ``'anomaly'``, and ``'incident'``.
        title : str, optional
            Chart title.
        xaxis_title : str, optional
            X-axis label.
        yaxis_title : str, optional
            Y-axis label.
        """
        if colors is None:
            colors = {'normal': 'blue', 'anomaly': 'red', 'incident': 'orange'}

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.anomaly_df.index, y=self.anomaly_df[self.score_col], mode='lines+markers',
            name='Anomaly Scores', line=dict(color=colors['normal']),
            marker=dict(size=5, opacity=0.4)
        ))

        fig.add_trace(go.Scatter(
            x=self.anomaly_df[self.anomaly_df[self.anomaly_col] == -1].index,
            y=self.anomaly_df.loc[self.anomaly_df[self.anomaly_col] == -1, self.score_col],
            mode='markers', name='Anomalies',
            marker=dict(color=colors['anomaly'], size=8)
        ))

        if threshold is not None:
            fig.add_hline(y=threshold, line_dash='dash', line_color='orange',
                          annotation_text=f'Threshold ({threshold:.3f})')

        if self.incidents_df is not None:
            for _, row in self.incidents_df.iterrows():
                fig.add_vrect(
                    x0=row['start_time'], x1=row['end_time'],
                    fillcolor=colors['incident'], opacity=0.1,
                    annotation_text=row['Servicio'],
                )

        fig.update_layout(
            title=title or 'Anomaly Scores',
            xaxis_title=xaxis_title or 'Time',
            yaxis_title=yaxis_title or 'Anomaly Score',
            legend=dict(x=0.01, y=0.99)
        )

        fig.show()

    def plot_score_distribution(self, bins=40, threshold=None, title=None):
        """Plot a histogram of anomaly scores split by normal/anomaly class.

        This visualization helps assess the **separation** between normal
        and anomalous score distributions. Good separation (little overlap)
        means the detector can reliably distinguish anomalies.

        Parameters
        ----------
        bins : int, default=40
            Number of histogram bins.
        threshold : float, optional
            If provided, draws a vertical threshold line.
        title : str, optional
            Chart title.
        """
        scores = self.anomaly_df[self.score_col]
        labels = self.anomaly_df[self.anomaly_col]

        normal_scores = scores[labels != -1]
        anomaly_scores = scores[labels == -1]

        plt.figure(figsize=(10, 5))
        plt.hist(normal_scores, bins=bins, alpha=0.6, label='Normal', color='steelblue', edgecolor='white')
        if len(anomaly_scores) > 0:
            plt.hist(anomaly_scores, bins=max(bins // 4, 5), alpha=0.8, label='Anomaly', color='red', edgecolor='white')

        if threshold is not None:
            plt.axvline(x=threshold, color='orange', linestyle='--', linewidth=1.5,
                        label=f'Threshold ({threshold:.3f})')

        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        plt.title(title or 'Score Distribution — Normal vs Anomaly')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_features(self, feature_columns=None, title=None):
        """Plot original features with anomaly markers overlaid.

        Shows the raw feature values over time with red markers on
        samples classified as anomalies. Useful for understanding
        *what happened* in the data when an anomaly was detected.

        Parameters
        ----------
        feature_columns : list of str, optional
            Columns to plot. Defaults to all numeric columns except
            the score and anomaly columns.
        title : str, optional
            Overall chart title.
        """
        df = self.anomaly_df
        exclude = {self.score_col, self.anomaly_col}

        if feature_columns is None:
            feature_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

        if not feature_columns:
            print("No numeric feature columns found to plot.")
            return

        n = len(feature_columns)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]

        anomaly_mask = df[self.anomaly_col] == -1

        for ax, col in zip(axes, feature_columns):
            ax.plot(df.index, df[col], linewidth=0.6, color='steelblue', alpha=0.7)
            if anomaly_mask.any():
                ax.scatter(df.index[anomaly_mask], df[col][anomaly_mask],
                           color='red', s=20, zorder=5, label='Anomaly')
            ax.set_ylabel(col)
            if ax is axes[0]:
                ax.legend(loc='upper right', fontsize=8)

        axes[-1].set_xlabel('Time')
        plt.suptitle(title or 'Feature Values with Anomaly Markers')
        plt.tight_layout()
        plt.show()


class SHAPVisualizer:
    """SHAP-based model interpretability visualizer.

    Explains predictions of tree-based detectors (e.g. ``IsolationForestDetector``)
    using SHAP values.

    Parameters
    ----------
    model : object
        A trained detector with a ``.model`` attribute that is a
        scikit-learn tree-based estimator.
    """

    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(self.model.model)
        self._shap_values_cache = None
        self._shap_X_cache = None

    @property
    def _base_value_scalar(self):
        """Return expected_value as a Python float (handles array case)."""
        ev = self.explainer.expected_value
        return float(ev[0]) if hasattr(ev, '__len__') and len(ev) == 1 else float(ev)

    def _get_shap_values(self, X):
        """Compute and cache SHAP values."""
        if self._shap_X_cache is not X:
            self._shap_values_cache = self.explainer.shap_values(X)
            self._shap_X_cache = X
        return self._shap_values_cache

    def plot_force(self, X, anomaly_index):
        """Plot a SHAP force plot for a single sample.

        Shows how each feature pushed the prediction from the base
        value toward the final output for the given sample.

        Parameters
        ----------
        X : pd.DataFrame
            Input samples (same data used for prediction).
        anomaly_index : int
            Row index in *X* to explain.
        """
        shap_values = self._get_shap_values(X)
        shap.force_plot(
            self.explainer.expected_value,
            shap_values[anomaly_index],
            X.iloc[anomaly_index],
            matplotlib=True,
            figsize=(20, 5),
            text_rotation=70
        )

    def plot_summary(self, X):
        """Plot a SHAP summary (beeswarm) plot.

        Each dot is one sample. The X axis shows the SHAP value
        (impact on prediction), color indicates the feature value
        (red = high, blue = low). Features are sorted by importance.

        Parameters
        ----------
        X : pd.DataFrame
            Input samples.
        """
        shap_values = self._get_shap_values(X)
        shap.summary_plot(shap_values, X)

    def plot_waterfall(self, X, sample_index):
        """Plot a SHAP waterfall chart for a single sample.

        A vertical bar chart showing each feature's contribution
        stacked from the base value to the final prediction.
        More readable than the force plot when there are many features.

        Parameters
        ----------
        X : pd.DataFrame
            Input samples.
        sample_index : int
            Row index in *X* to explain.
        """
        shap_values = self._get_shap_values(X)
        explanation = shap.Explanation(
            values=shap_values[sample_index],
            base_values=self._base_value_scalar,
            data=X.iloc[sample_index].values,
            feature_names=list(X.columns),
        )
        try:
            shap.plots.waterfall(explanation)
        except (IndexError, ValueError):
            # Some SHAP versions fail when the waterfall plot receives a scalar
            # base value for a single sample. Fallback to the bar chart so the
            # notebook remains usable without changing the API.
            shap.plots.bar(explanation)

    def plot_bar(self, X):
        """Plot a SHAP bar chart of global feature importance.

        Shows the mean absolute SHAP value per feature as horizontal
        bars. Cleaner than the summary plot when you only need the
        importance ranking without the distribution detail.

        Parameters
        ----------
        X : pd.DataFrame
            Input samples.
        """
        shap_values = self._get_shap_values(X)
        explanation = shap.Explanation(
            values=shap_values,
            base_values=np.full(len(X), self._base_value_scalar),
            data=X.values,
            feature_names=list(X.columns),
        )
        shap.plots.bar(explanation)

    def plot_dependence(self, X, feature, interaction_feature=None):
        """Plot a SHAP dependence plot for a single feature.

        Scatter plot of a feature's value vs its SHAP value, optionally
        colored by an interaction feature. Reveals non-linear relationships
        (e.g. "CPU only matters when it exceeds 80%") and feature interactions.

        Parameters
        ----------
        X : pd.DataFrame
            Input samples.
        feature : str
            Feature name to plot on the X axis.
        interaction_feature : str or None, optional
            Feature to use for coloring. If ``None``, SHAP picks the
            strongest interaction automatically.
        """
        shap_values = self._get_shap_values(X)
        shap.dependence_plot(
            feature, shap_values, X,
            interaction_index=interaction_feature,
        )
