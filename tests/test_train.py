# tests/test_train.py  (replace entire file)
import pandas as pd
import numpy as np
from pathlib import Path
import sys

from src.training import train

def make_dummy_df(tmp_path, n_per_class=6):
    """Create a small but safe dataframe (2 classes, plenty of rows)."""
    n = 2 * n_per_class
    data = {
        "file_name": [f"f{i}.wav" for i in range(n)],
        **{f"mfcc_{j+1}": np.random.randn(n) for j in range(40)},
        "class": ["dog"] * n_per_class + ["cat"] * n_per_class,
    }
    df = pd.DataFrame(data)
    out = tmp_path / "fold1_df_file.csv"
    df.to_csv(out, index=False)
    return tmp_path

def test_split_and_encode(tmp_path):
    folder = make_dummy_df(tmp_path, n_per_class=6)  # 12 rows total
    df = train.load_archive_dataframes(str(folder))
    X, y = train.split_xy(df, "class")
    # Use gentler split sizes to avoid tiny folds
    X_tr, X_va, X_te, y_tr, y_va, y_te = train.make_splits(
        X, y, test_size=0.2, val_size=0.2, seed=42
    )
    assert X_tr.shape[1] == 40
    # all splits non-empty
    assert len(y_tr) > 0 and len(y_va) > 0 and len(y_te) > 0
    # preserve both classes across splits
    assert set(y_tr) | set(y_va) | set(y_te) == {"dog", "cat"}

def test_main_runs(tmp_path):
    folder = make_dummy_df(tmp_path, n_per_class=10)
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    sys.argv = [
        "train.py",
        "--archive_df_dir", str(folder),
        "--target", "class",
        "--models_dir", str(models_dir),
        "--reports_dir", str(reports_dir),
        "--pca_components", "5",
        "--test_size", "0.2",
        "--val_size", "0.2",
        "--seed", "42",
    ]
    train.main()

    # Artifacts
    assert (models_dir / "model.pkl").exists()
    assert (models_dir / "pca.pkl").exists()
    assert (models_dir / "scaler.pkl").exists()
    assert (models_dir / "label_encoder.pkl").exists()
    assert (reports_dir / "metrics.csv").exists()
