import pandas as pd
from pathlib import Path
import pytest

dio = pytest.importorskip("src.training.data_io", reason="src.training.data_io not found")

def test_save_and_load_csv_roundtrip(tmp_path: Path):
    df = pd.DataFrame({"a":[1,2], "b":[3,4]})
    p = tmp_path / "x.csv"
    assert hasattr(dio, "save_csv"), "data_io.save_csv is missing"
    assert hasattr(dio, "load_csv"), "data_io.load_csv is missing"
    dio.save_csv(df, p)
    assert p.exists()
    df2 = dio.load_csv(p)
    pd.testing.assert_frame_equal(df, df2)
