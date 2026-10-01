import numpy as np

from sentinel.detectors import IsolationForestDetector


def test_predict_proba_single_sample_is_finite():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(128, 2))

    detector = IsolationForestDetector(
        n_estimators=40,
        random_state=42,
    )
    detector.fit(X)

    sample = np.array([[2.0, 2.0]])
    proba = detector.predict_proba(sample)

    assert proba.shape == (1,)
    assert np.isfinite(proba).all()
    assert 0.0 <= proba[0] <= 1.0


def test_predict_proba_is_stable_across_prediction_batches():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(128, 2))

    detector = IsolationForestDetector(
        n_estimators=40,
        random_state=42,
    )
    detector.fit(X)

    batch_a = np.array([
        [2.0, 2.0],
        [0.0, 0.0],
    ])
    batch_b = np.array([
        [2.0, 2.0],
        [100.0, 100.0],
    ])

    proba_a = detector.predict_proba(batch_a)
    proba_b = detector.predict_proba(batch_b)

    assert np.allclose(proba_a[0], proba_b[0])
    assert not np.isnan(proba_a[0])
    assert not np.isnan(proba_b[0])
