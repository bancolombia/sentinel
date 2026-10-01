import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rrcf")

from sentinel.detectors import AutoencoderDetector, IsolationForestDetector, RRCFDetector
from sentinel.ingestion.base_parser import BaseLogParser


class CSVParser(BaseLogParser):
    def parse(self):
        import pandas as pd

        return pd.read_csv(self.file_path)


def test_public_detector_api_smoke(tmp_path):
    X = np.random.randn(30, 3)

    iso = IsolationForestDetector(n_estimators=10, random_state=0)
    iso.fit(X)
    assert iso.predict(X).shape == (30,)
    assert np.isfinite(iso.predict_proba(X)).all()

    rrcf = RRCFDetector(num_trees=5, tree_size=12, shingle_size=1)
    scores = rrcf.fit_predict(X[:, 0])
    assert scores.shape == (30,)
    assert np.isfinite(scores).all()

    ae = AutoencoderDetector(
        n_features=3,
        seq_len=1,
        latent_dim=4,
        learning_rate=0.01,
        batch_size=8,
        epochs=2,
        threshold_multiplier=3.0,
    )
    ae.fit(X)
    model_path = tmp_path / "detector.pt"
    ae.save_model(str(model_path))

    loaded = AutoencoderDetector(
        n_features=3,
        seq_len=1,
        latent_dim=4,
        learning_rate=0.01,
        batch_size=8,
        epochs=2,
        threshold_multiplier=3.0,
    )
    loaded.load_model(str(model_path))

    assert loaded.threshold is not None
    assert loaded.predict(X[:5]).shape == (5,)


def test_ingestion_parser_happy_path(tmp_path):
    csv_path = tmp_path / "logs.csv"
    csv_path.write_text(
        "timestamp,level,message\n2025-01-01 00:00:00,INFO,ok\n2025-01-01 00:00:01,ERROR,fail\n",
        encoding="utf-8",
    )

    parser = CSVParser(str(csv_path))
    df = parser.parse()

    assert list(df.columns) == ["timestamp", "level", "message"]
    assert len(df) == 2
