import numpy as np
import pytest

pytest.importorskip("torch")

from sentinel.detectors import AutoencoderDetector


def test_autoencoder_save_load_restores_scaler_and_threshold(tmp_path):
    X = np.random.randn(30, 15)
    detector = AutoencoderDetector(
        n_features=3,
        seq_len=5,
        latent_dim=4,
        learning_rate=0.01,
        batch_size=8,
        epochs=2,
        threshold_multiplier=3.0,
    )
    detector.fit(X)

    model_path = tmp_path / "autoencoder_model.pt"
    detector.save_model(str(model_path))

    restored = AutoencoderDetector(
        n_features=3,
        seq_len=5,
        latent_dim=4,
        learning_rate=0.01,
        batch_size=8,
        epochs=2,
        threshold_multiplier=3.0,
    )
    restored.load_model(str(model_path))

    assert restored.threshold is not None
    assert hasattr(restored.scaler, "mean_")
    prediction = restored.predict(X[:5])
    assert prediction.shape == (5,)
