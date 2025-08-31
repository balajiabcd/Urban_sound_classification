import numpy as np
import pandas as pd
from pathlib import Path

def test_list_audio_files(tmp_path):
    from extract import extract_data as ed
    # create fake audio files
    for name in ["a.wav", "b.wav", "c.mp3", "ignore.txt"]:
        (tmp_path / name).write_bytes(b"fake")
    files = ed.list_audio_files(tmp_path)
    names = sorted([p.name for p in files])
    assert names == ["a.wav", "b.wav", "c.mp3"]

def test_extract_features_for_file(tmp_path, mock_librosa):
    from extract import extract_data as ed
    p = tmp_path / "a.wav"
    p.write_bytes(b"fake")
    feats = ed.extract_features_for_file(p, n_mfcc=40)
    # Should be a flat dict or 1D vector-like with 40*n_stats elements
    if isinstance(feats, dict):
        assert len(feats) >= 40
    else:
        arr = np.asarray(feats).ravel()
        assert arr.ndim == 1 and arr.size >= 40

def test_build_dataframe_from_folder(tmp_path, mock_librosa):
    from extract import extract_data as ed
    # create a few fake files
    for i in range(3):
        (tmp_path / f"f{i}.wav").write_bytes(b"fake")
    df = ed.build_dataframe_from_folder(tmp_path, n_mfcc=20)
    assert isinstance(df, pd.DataFrame) and len(df) == 3
