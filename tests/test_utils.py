import os
import numpy as np
import pandas as pd
import pytest
from src.extract import utils

def test_get_features(tmp_path):
    # Create a dummy wav file
    import soundfile as sf
    file_path = tmp_path / "test.wav"
    sf.write(file_path, np.random.randn(8000), 8000)

    features = utils.get_features(str(file_path))
    assert isinstance(features, list)
    assert len(features) == 40

def test_make_dataframe(tmp_path):
    # Dummy wav + metadata
    import soundfile as sf
    file_path = tmp_path / "a.wav"
    sf.write(file_path, np.random.randn(8000), 8000)
    class_map = {"a.wav": "dog"}

    df = utils.make_dataframe(str(tmp_path), class_map)
    assert "file_name" in df.columns
    assert "class" in df.columns
    assert df.iloc[0]["class"] == "dog"
