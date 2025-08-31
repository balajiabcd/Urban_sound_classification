import numpy as np
import pandas as pd

def test_split_train_valid_stratified():
    from training import preprocess as pp
    X = np.random.randn(100, 10)
    y = np.array([0]*50 + [1]*50)
    X_tr, X_va, y_tr, y_va = pp.split_train_valid(X, y, test_size=0.2, random_state=0, stratify=True)
    assert len(y_va) == 20 and set(np.unique(y_tr)) == {0,1}

def test_fit_transformers_and_apply(tmp_path):
    from training import preprocess as pp
    X = np.random.randn(50, 12)
    obj = pp.fit_transformers(X, n_components=6, save_dir=tmp_path)
    # Expect attributes like scaler_ and pca_
    X2 = pp.apply_transformers(X, obj)
    assert X2.shape[1] in (6, X.shape[1])
