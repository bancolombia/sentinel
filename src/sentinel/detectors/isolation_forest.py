from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator
import numpy as np

class IsolationForestDetector(BaseEstimator):
    """Anomaly detector based on the Isolation Forest algorithm.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of base estimators in the ensemble.
    max_samples : int, float or 'auto', default='auto'
        Samples to draw for each base estimator.
    contamination : 'auto' or float, default='auto'
        Expected proportion of outliers in the dataset.
    random_state : int, RandomState or None, default=None
        Controls randomness for reproducibility.
    """
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto', random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state
        )

    def fit(self, X, y=None):
        """
        Fits the Isolation Forest model to the provided data.

        Args:
        
            - X (array-like or sparse matrix): The input data to fit the model. 
                It should be of shape (n_samples, n_features).

            - y (ignored): Not used, present here for API consistency by convention.

        Returns:

            self: Returns the instance of the detector after fitting the model.
        """
        self.model.fit(X)
        return self
        
    def predict(self, X):
        """
        Predicts the anomaly labels for the given input data using the Isolation Forest model.

        Args:

             X (array-like or sparse matrix): The input data to predict. It should have the same 
                                             number of features as the data used for training the model.

        Returns:

            array: An array of anomaly labels for each data point in X. The output is -1 for anomalies 
                   and 1 for normal points.
        """
        return self.model.predict(X)
        
    def decision_function(self, X):
        """
        Compute the anomaly score for the input data.

        Args:

            X (array-like or sparse matrix): The input samples to evaluate.

        Returns:

            numpy.ndarray: The anomaly scores for each input sample. Lower scores
            indicate more anomalous samples.
        """
        return self.model.decision_function(X)
        
    def fit_predict(self, X, y=None):
        """
        Fits the Isolation Forest model to the data and predicts anomalies.

        Args:

            - X (array-like or sparse matrix): The input data to fit and predict. 
                It should be of shape (n_samples, n_features).

            - y (ignored): Not used, present for API consistency by convention.

        Returns:

            array: The anomaly scores for each sample. 
                -1 indicates an anomaly, and 1 indicates normal data.
        """
        return self.model.fit_predict(X)
        
    def get_anomalies(self, X):
        """
        Get the anomalies in the dataset X.
        
        Args
        
            X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        
            anomalies : array-like of shape (n_anomalies, n_features)
            The anomalies found in the dataset.
        """
        predictions = self.model.predict(X)
        anomalies = X[predictions == -1]
        return anomalies
    
    def predict_proba(self, X):
        """
        Predict the probability of each sample being an anomaly.

        The raw Isolation Forest decision score is already a calibrated measure
        of how anomalous a sample is. A batch-wise min-max normalization makes the
        result depend on the composition of the current prediction batch, which is
        undesirable for a fitted detector. We convert the score to a stable
        anomaly probability using a logistic transform that is independent of the
        surrounding batch.

        Args
        
            X : array-like of shape (n_samples, n_features)
                The input samples.

        Returns
        
            proba : array-like of shape (n_samples,)
                The probability of each sample being an anomaly.
        """
        decision_scores = np.asarray(self.model.decision_function(X), dtype=float)
        if decision_scores.ndim == 0:
            decision_scores = decision_scores.reshape(1)

        # Lower scores indicate more anomalous observations. Mapping the
        # decision score through a sigmoid keeps the probability bounded in
        # [0, 1] without depending on the current batch composition.
        logits = np.clip(-decision_scores, -500.0, 500.0)
        proba = 1.0 / (1.0 + np.exp(logits))
        return proba

if __name__ == "__main__":
    # Example usage
    X_train = np.random.rand(100, 2)
    X_test = np.random.rand(20, 2)

    detector = IsolationForestDetector(n_estimators=100, random_state=42)
    detector.fit(X_train)
    predictions = detector.predict(X_test)
    scores = detector.decision_function(X_test)
    anomalies = detector.get_anomalies(X_test)
    proba = detector.predict_proba(X_test)

    print("Predictions:", predictions)
    print("Scores:", scores)
    print("Anomalies:", anomalies)
    print("Anomaly Probabilities:", proba)