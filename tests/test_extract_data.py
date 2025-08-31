import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

ed = pytest.importorskip("src.extract.extract_data", reason="src.extract.extract_data not found")

@pytest.fixture(autouse=True)
def mock_librosa(monkeypatch):
    # Mock librosa to avoid heavy audio deps
    def _load(path, sr=None):
        return np.zeros(22050), 22050
    def _mfcc(y, sr, n_mfcc=40):
        return np.tile(np.arange(n_mfcc, dtype=float).reshape(-1,1), (1,10))
    monkeypatch.setitem(ed.__dict__, "librosa", SimpleNamespace(load=_load, feature=SimpleNamespace(mfcc=_mfcc)))

def test_list_audio_files(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.mp3").write_bytes(b"")
    (tmp_path / "ignore.txt").write_text("x")
    assert hasattr(ed, "list_audio_files"), "extract_data.list_audio_files missing"
    files = ed.list_audio_files(tmp_path)
    assert set(p.suffix for p in files) <= {".wav",".mp3",".flac"}
    assert len(files) == 2

def test_extract_features_for_file(tmp_path):
    f = tmp_path / "a.wav"; f.write_bytes(b"")
    assert hasattr(ed, "extract_features_for_file"), "extract_data.extract_features_for_file missing"
    feat = ed.extract_features_for_file(f, n_mfcc=40)
    assert isinstance(feat, np.ndarray)
    assert feat.shape[0] in (40, )  # mean/or flattened MFCCs

def test_build_dataframe_from_folder(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.wav").write_bytes(b"")
    assert hasattr(ed, "build_dataframe_from_folder"), "extract_data.build_dataframe_from_folder missing"
    df = ed.build_dataframe_from_folder(tmp_path, n_mfcc=40)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
