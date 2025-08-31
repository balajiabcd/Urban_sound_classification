from pathlib import Path
import pandas as pd

def _load_model(path: Path):
    path = Path(path)
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

def predict_from_artifacts(model_path, X: pd.DataFrame):
    """
    Load a persisted sklearn model and run predictions on X.
    """
    model = _load_model(model_path)
    return model.predict(X)
