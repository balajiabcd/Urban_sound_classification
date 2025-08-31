import numpy as np
import pandas as pd
import pytest
from pathlib import Path

pr = pytest.importorskip("src.training.predict", reason="src.training.predict not found")
models = pytest.importorskip("src.training.models", reason="src.training.models not found")

def test_predict_pipeline_from_artifacts(tmp_path: Path):
    X = pd.DataFrame(np.random.randn(60, 5), columns=list("ABCDE"))
    y = np.array([0]*30 + [1]*30)
    clf = models.get_model(next(iter(models.get_model_known_names())))
    clf.fit(X, y)

    model_path = tmp_path / "model.pkl"
    try:
        import joblib; joblib.dump(clf, model_path)
    except Exception:
        import pickle
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

    assert hasattr(pr, "predict_from_artifacts"), "predict.predict_from_artifacts missing"
    yp = pr.predict_from_artifacts(model_path, X.head())
    assert len(yp) == len(X.head())
