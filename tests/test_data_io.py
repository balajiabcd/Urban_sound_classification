import pandas as pd
import tempfile
from src.training import data_io

def test_load_archive_dataframes(tmp_path):
    f1 = tmp_path / "a.csv"
    pd.DataFrame({"x":[1,2]}).to_csv(f1, index=False)
    f2 = tmp_path / "b.csv"
    pd.DataFrame({"x":[3,4]}).to_csv(f2, index=False)

    df = data_io.load_archive_dataframes(str(tmp_path))
    assert len(df) == 4
