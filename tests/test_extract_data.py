# tests/test_extract_data.py  (fixed)
import pandas as pd
from src.extract import extract_data

def test_class_by_file_name():
    meta = pd.DataFrame({"slice_file_name": ["a.wav"], "class": ["dog"]})
    mapping = meta.set_index("slice_file_name")["class"].to_dict()
    assert mapping["a.wav"] == "dog"
