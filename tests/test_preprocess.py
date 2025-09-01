import pandas as pd
import numpy as np
from src.training import preprocess

def test_split_xy():
    df = pd.DataFrame({
        "file": ["a","b"],
        "mfcc_1": [0.1,0.2],
        "mfcc_2": [0.3,0.4],
        "class": ["dog","cat"]
    })
    X,y = preprocess.split_xy(df, "class")
    assert X.shape[1] == 2
    assert list(y) == ["dog","cat"]

def test_make_splits():
    X = pd.DataFrame(np.random.randn(50,5))
    y = pd.Series(["a"]*25 + ["b"]*25)
    X_tr, X_va, X_te, y_tr, y_va, y_te = preprocess.make_splits(X,y)
    assert len(X_tr)+len(X_va)+len(X_te) == 50

def test_pca_scaler():
    X = np.random.randn(20,10)
    y = np.random.choice(["a","b"],20)
    X_tr,X_va,X_te,y_tr,y_va,y_te = preprocess.make_splits(X,y)
    Xt, Xv, Xtt, pca, sc = preprocess.fit_transform_pca_scaler(X_tr,X_va,X_te, n_components=5)
    assert Xt.shape[1] == 5
