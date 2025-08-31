import numpy as np
import pytest
ev = pytest.importorskip("src.training.evaluate", reason="src.training.evaluate not found")

def test_get_metrics_shapes():
    y_true = np.array([0,1,1,0,1,0])
    y_pred = np.array([0,1,0,0,1,1])
    assert hasattr(ev, "get_metrics"), "evaluate.get_metrics is missing"
    m = ev.get_metrics("rf", y_true, y_pred)
    # tests expect lowercase keys
    for k in {"accuracy","precision","recall","f1"}:
        assert k in m, f"missing key: {k} in metrics: {list(m.keys())}"
    assert 0.0 <= m["accuracy"] <= 1.0
