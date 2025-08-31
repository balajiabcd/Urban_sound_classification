import numpy as np
import pandas as pd

def test_train_pipeline_end_to_end(tmp_path):
    from training import train as tr
    # minimal fake dataframe
    X = np.random.randn(60, 10)
    y = np.array([0]*30 + [1]*30)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["target"] = y
    out = tr.train_pipeline(
        df=df,
        target_col="target",
        model_name="rf",
        artifacts_dir=tmp_path,
        n_components=5,
        random_state=0
    )
    # expect model + transformers saved
    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "transformers.pkl").exists()
    assert "metrics" in out and "model_path" in out
