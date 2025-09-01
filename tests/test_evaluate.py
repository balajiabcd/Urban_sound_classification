import numpy as np
from src.training import evaluate

def test_get_metrics():
    y_true = [0,1,1,0]
    y_pred = [0,1,0,0]
    m = evaluate.get_metrics("dummy", y_true, y_pred)
    assert "Accuracy" in m
    assert "F1_Score" in m
