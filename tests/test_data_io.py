import pandas as pd

def test_save_and_load_csv_roundtrip(tmp_path):
    from training import data_io as dio
    df = pd.DataFrame({"a":[1,2], "b":[3,4]})
    p = tmp_path / "x.csv"
    dio.save_csv(df, p)
    df2 = dio.load_csv(p)
    pd.testing.assert_frame_equal(df, df2)
