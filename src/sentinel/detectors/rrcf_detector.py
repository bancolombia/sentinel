"""Robust Random Cut Forest anomaly detection implementation."""
import numpy as np
import pandas as pd
import rrcf
from sklearn.base import BaseEstimator


class RRCFDetector(BaseEstimator):

    """    RRCFDetector: Robust Random Cut Forest anomaly detector.

    This class implements anomaly detection using the Robust Random Cut Forest 
    algorithm for streaming anomaly detection.

    Args:

        - shingle_size: Size of the shingling window.

        - num_trees: Number of trees in the random forest.

        - tree_size: Maximum size of each tree in the forest.

    
    Attributes:

        - shingle_size : int
            Size of the shingling window.

        - num_trees : int
            Number of trees in the random forest.

        - tree_size : int
            Maximum size of each tree in the forest.

        - forest : list
            List of RCTree objects representing the forest.

        - index : None or int
            Index used for tracking data points (if applicable).

        - anomaly_scores : None or pd.Series
            Series containing anomaly scores for the input data.
    """

    def __init__(self, shingle_size=15, num_trees=100, tree_size=500):
        self.shingle_size = shingle_size
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.forest = [rrcf.RCTree() for _ in range(self.num_trees)]
        self.index = None
        self.anomaly_scores = None

    def _as_series(self, X):
        """Normalize array-like inputs to a pandas Series for downstream RRCF operations."""
        if isinstance(X, pd.Series):
            return X.reset_index(drop=True)
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("RRCFDetector expects a 1D series-like input.")
            return X.iloc[:, 0].reset_index(drop=True)
        values = np.asarray(X, dtype=float)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim > 1:
            raise ValueError("RRCFDetector expects a 1D array-like input.")
        return pd.Series(values)

    def fit(self, X, y=None):
        """Fit the model to the input data.

        Args
        
            X : array-like of shape (n_samples,)
                Input time series data.

        Returns
        
            self : object
                Returns self.
        """
        X = self._as_series(X)
        n = len(X)
        self.anomaly_scores = pd.Series(0.0, index=np.arange(n))
        
        for tree in self.forest:
            for i in range(n):
                point = X.iloc[i]
                unique_index = (i, id(tree))
                tree.insert_point(point, index=unique_index)
                
                if len(tree.leaves) > self.tree_size:
                    old_index = (i - self.tree_size, id(tree))
                    if old_index in tree.leaves:
                        tree.forget_point(old_index)
        
        return self

    def predict(self, X):
        """Calculate anomaly scores for input data.

        Args
        
            X : array-like of shape (n_samples,)
                Input time series data.

        Returns
        
            scores : ndarray of shape (n_samples,)
                Anomaly scores for each input sample.
        """
        X = self._as_series(X)
        scores = np.zeros(len(X))
        
        for tree in self.forest:
            for i in range(len(X)):
                unique_index = (i, id(tree))
                
                if unique_index in tree.leaves:
                    leaf = tree.leaves[unique_index]
                    score = tree.codisp(leaf)
                    scores[i] += score
        
        return scores / self.num_trees  # Normalize by number of trees

    def fit_predict(self, X, y=None):
        """Fit the model and calculate anomaly scores.

        Args
        
            X : array-like of shape (n_samples,)
                Input time series data.

        Returns
        
            anomaly_scores : Series of shape (n_samples,)
                Anomaly scores for each input sample.
        """
        self.fit(X)
        self.anomaly_scores = pd.Series(self.predict(X), index=np.arange(len(X)))
        return self.anomaly_scores

    def get_anomalies(self, threshold):
        """Identify anomalies based on a threshold.

        Args
        
            threshold : float
                Threshold value for anomaly detection.

        Returns
        
            anomalies : Series
                Series containing anomaly scores above the threshold.
        """
        if self.anomaly_scores is None:
            raise ValueError("Model must be fit before getting anomalies.")
        return self.anomaly_scores[self.anomaly_scores > threshold]

    def predict_proba(self, X):
        """Calculate normalized probability-like anomaly scores.

        Args
        
            X : array-like of shape (n_samples,)
                Input time series data.

        Returns
        
            proba : ndarray of shape (n_samples,)
                Normalized anomaly scores.
        """
        scores = self.predict(X)
        max_score = np.max(scores)
        
        # Avoid division by zero
        if max_score > 0:
            return scores / max_score
        return scores


# Example usage:
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(0)
    X = np.random.randn(1000)

    # Initialize the detector
    detector = RRCFDetector()

    # Get anomaly scores
    anomaly_scores = detector.fit_predict(X)

    # Get anomalies with a threshold
    anomalies = detector.get_anomalies(threshold=3.0)

    # Print the results
    print("Anomaly Scores:\n", anomaly_scores)
    print("Anomalies:\n", anomalies)