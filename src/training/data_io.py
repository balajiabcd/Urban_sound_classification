from pathlib import Path
import pandas as pd

def load_archive_dataframes(folder: str) -> pd.DataFrame:
    p = Path(folder)
    csvs = sorted([x for x in p.iterdir() if x.suffix.lower()==".csv"])
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {folder}")
    df = pd.read_csv(csvs[0])
    for f in csvs[1:]:
        df = pd.concat([df, pd.read_csv(f)], ignore_index=True)
    return df
