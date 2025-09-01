# tests/test_predict.py  (replace entire file)
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Use the real entrypoint so we exercise your CLI
from src.training import predict

class DummyPCA:
    def transform(self, X):
        # identity transform
        return X

class DummyScaler:
    def transform(self, X):
        # identity transform
        return X

class DummyLabelEncoder:
    def inverse_transform(self, y):
        # map 0->"dog", 1->"cat"
        mapping = {0: "dog", 1: "cat"}
        return np.array([mapping[int(v)] for v in y])

class DummyModel:
    def predict(self, X):
        # predict class 0 for all rows
        return np.zeros(len(X), dtype=int)

def test_predict_pipeline(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Save dummy artifacts
    joblib.dump(DummyModel(), models_dir / "model.pkl")
    joblib.dump(DummyPCA(),   models_dir / "pca.pkl")
    joblib.dump(DummyScaler(), models_dir / "scaler.pkl")
    joblib.dump(DummyLabelEncoder(), models_dir / "label_encoder.pkl")

    # Make a CSV with numeric feature columns (like your X at inference)
    features_csv = tmp_path / "features.csv"
    pd.DataFrame({
        "file_name": ["a","b","c"],  # non-numeric should be ignored by select_dtypes
        "mfcc_1": [0.1, 0.2, 0.3],
        "mfcc_2": [0.4, 0.5, 0.6],
    }).to_csv(features_csv, index=False)

    out_csv = tmp_path / "preds.csv"

    # Simulate CLI args
    sys.argv = [
        "predict.py",
        "--models_dir", str(models_dir),
        "--features_csv", str(features_csv),
        "--out_csv", str(out_csv),
    ]
    predict.main()

    # Check predictions
    df = pd.read_csv(out_csv)
    assert list(df["prediction"]) == ["dog", "dog", "dog"]
