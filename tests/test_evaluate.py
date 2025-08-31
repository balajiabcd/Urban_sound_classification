import numpy as np

def test_get_metrics_shapes():
    from training import evaluate as ev
    y_true = np.array([0,1,1,0,1,0])
    y_pred = np.array([0,1,0,0,1,1])
    metrics = ev.get_metrics("rf", y_true, y_pred)
    assert {"accuracy","precision","recall","f1"} <= set(metrics)
