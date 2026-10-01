import numpy as np

from sentinel.detectors import RRCFDetector


def test_rrcf_accepts_numpy_array_like_input():
    X = np.random.randn(50)
    detector = RRCFDetector(num_trees=4, tree_size=16, shingle_size=1)

    fitted = detector.fit(X)
    scores = detector.predict(X)

    assert fitted is detector
    assert scores.shape == X.shape
    assert np.isfinite(scores).all()
