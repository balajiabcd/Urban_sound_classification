from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_metrics(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Return common classification metrics with LOWERCASE keys, as expected by tests.
    """
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
