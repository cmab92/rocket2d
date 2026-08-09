import numpy as np
import torch

from rocket2d.models.cnn import SimpleCNN
from rocket2d.models.rocket import RocketClassifier


def test_simple_cnn_forward_shape():
    model = SimpleCNN(num_classes=3, img_size=32)
    x = torch.randn(4, 1, 32, 32)
    out = model(x)
    assert out.shape == (4, 3)


def test_rocket_classifier_fit_predict():
    # Image size must be large enough for the largest kernel/dilation combo
    # (default k up to 9, d up to 2 -> effective span (9-1)*2+1 = 17).
    rng = np.random.default_rng(0)
    X = rng.random((40, 20, 20), dtype=np.float32)
    y = (np.arange(40) % 2).astype(np.int64)

    model = RocketClassifier(n_kernels=50, seed=0, batch_size=8)
    model.fit(X, y)
    assert model.is_fitted

    preds = model.predict(X)
    assert preds.shape == (40,)

    acc = model.score(X, y)
    assert 0.0 <= acc <= 1.0


def test_rocket_classifier_predict_before_fit_raises():
    import pytest

    model = RocketClassifier(n_kernels=10)
    X = np.zeros((2, 8, 8), dtype=np.float32)
    with pytest.raises(RuntimeError):
        model.predict(X)
