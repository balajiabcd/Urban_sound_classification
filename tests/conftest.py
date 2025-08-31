import builtins
import json
import types
import numpy as np
import pandas as pd
import pytest

# tests/conftest.py
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)



@pytest.fixture
def rng():
    return np.random.default_rng(0)

@pytest.fixture
def tiny_classification_data(rng):
    # small, linearly-separable-ish dataset
    X0 = rng.normal(loc=-1, scale=0.2, size=(20, 8))
    X1 = rng.normal(loc=+1, scale=0.2, size=(20, 8))
    X = np.vstack([X0, X1])
    y = np.array([0]*20 + [1]*20)
    return X, y

@pytest.fixture
def tmp_csv(tmp_path, rng):
    df = pd.DataFrame({
        "feature_"+str(i): rng.normal(size=10) for i in range(5)
    })
    p = tmp_path / "tmp.csv"
    df.to_csv(p, index=False)
    return p

@pytest.fixture
def mock_librosa(monkeypatch, rng):
    """Mock minimal parts of librosa used during feature extraction."""
    m = types.SimpleNamespace()
    def load(path, sr=None, mono=True):
        # Generate a 1-second fake waveform at 22_050 Hz
        sr_out = 22050 if sr is None else sr
        y = rng.normal(scale=0.01, size=sr_out).astype(np.float32)
        return y, sr_out
    m.load = load

    def feature_mfcc(y=None, sr=22050, n_mfcc=40):
        # Return constant-shaped MFCCs (n_mfcc x frames)
        frames = max(1, len(y)//512)
        return rng.normal(size=(n_mfcc, frames)).astype(np.float32)
    # mimic librosa.feature.mfcc namespace
    mfeature = types.SimpleNamespace(mfcc=feature_mfcc)
    m.feature = mfeature

    monkeypatch.setitem(builtins.__dict__, "librosa", m)
    return m
