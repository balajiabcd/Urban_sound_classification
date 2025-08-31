from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from . import models as mdl
from . import preprocess as pp
from .evaluate import get_metrics

def train_pipeline(
    df: pd.DataFrame,
    target_col: str,
    model_name: str = "rf",
    artifacts_dir: Optional[Path] = None,
    n_components: int = None,  # kept for API compatibility; unused in this minimal version
    random_state: int = 0
) -> Dict[str, Any]:
    """
    End-to-end training pipeline:
      - split (stratified)
      - fit transformers (StandardScaler)
      - train model
      - evaluate on validation split
      - (optional) persist artifacts
      - return artifacts, splits, metrics, and saved paths
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_tr, X_va, y_tr, y_va = pp.split_train_valid_stratified(X, y, test_size=0.2, random_state=random_state)
    transformers = pp.fit_transformers(X_tr)
    X_tr_s = pp.apply_transformers(X_tr, **transformers)
    X_va_s = pp.apply_transformers(X_va, **transformers)

    clf = mdl.get_model(model_name)
    clf.fit(X_tr_s, y_tr)

    # Evaluate
    y_va_pred = clf.predict(X_va_s)
    metrics = get_metrics(model_name, y_va, y_va_pred)

    model_path = None
    transformers_path = None

    # optionally save artifacts
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifacts_dir / "model.pkl"
        transformers_path = artifacts_dir / "transformers.pkl"
        try:
            import joblib
            joblib.dump(clf, model_path)
            joblib.dump(transformers, transformers_path)
        except Exception:
            import pickle
            with open(model_path, "wb") as f:
                pickle.dump(clf, f)
            with open(transformers_path, "wb") as f:
                pickle.dump(transformers, f)

    return {
        "model": clf,
        "transformers": transformers,
        "X_train": X_tr_s,
        "X_valid": X_va_s,
        "y_train": y_tr,
        "y_valid": y_va,
        "metrics": metrics,
        "model_path": str(model_path) if model_path else None,
        "transformers_path": str(transformers_path) if transformers_path else None,
    }
