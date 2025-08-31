import numpy as np
import pytest

models = pytest.importorskip("src.training.models", reason="src.training.models not found")

def test_get_model_known_names():
    assert hasattr(models, "get_model_known_names"), "models.get_model_known_names missing"
    names = set(models.get_model_known_names())
    assert len(names) > 0
    assert any(n in names for n in {"rf","random_forest","svm","svc","logreg","knn"})

def test_model_fit_predict():
    assert hasattr(models, "get_model"), "models.get_model missing"
    X = np.random.randn(40, 5)
    y = np.array([0]*20 + [1]*20)
    clf = models.get_model(next(iter(models.get_model_known_names())))
    clf.fit(X, y)
    yp = clf.predict(X)
    assert yp.shape == y.shape
