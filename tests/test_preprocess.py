import numpy as np
import pandas as pd
import pytest

pp = pytest.importorskip("src.training.preprocess", reason="src.training.preprocess not found")

def test_split_train_valid_stratified():
    X = pd.DataFrame(np.random.randn(100, 4), columns=list("ABCD"))
    y = pd.Series([0]*50 + [1]*50)
    assert hasattr(pp, "split_train_valid_stratified"), "preprocess.split_train_valid_stratified missing"
    X_tr, X_va, y_tr, y_va = pp.split_train_valid_stratified(X, y, test_size=0.2, random_state=42)
    assert len(X_tr) + len(X_va) == len(X)
    assert y_tr.mean() == pytest.approx(y.mean(), abs=0.1)

def test_fit_transformers_and_apply(tmp_path):
    X = pd.DataFrame(np.random.randn(50, 6), columns=list("ABCDEF"))
    assert hasattr(pp, "fit_transformers"), "preprocess.fit_transformers missing"
    assert hasattr(pp, "apply_transformers"), "preprocess.apply_transformers missing"
    transformers = pp.fit_transformers(X)
    X2 = pp.apply_transformers(X.copy(), **transformers)
    assert X2.shape[0] == X.shape[0]
