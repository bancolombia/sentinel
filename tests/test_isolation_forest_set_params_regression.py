import numpy as np

from sentinel.detectors import IsolationForestDetector


def test_set_params_updates_fitted_estimator_configuration():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(128, 2))

    detector = IsolationForestDetector(
        n_estimators=100,
        random_state=42,
    )

    detector.set_params(n_estimators=5)
    detector.fit(X)

    assert detector.get_params()["n_estimators"] == 5
    assert detector.model.n_estimators == 5
    assert len(detector.model.estimators_) == 5
