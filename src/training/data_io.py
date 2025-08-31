from pathlib import Path
from typing import Union
import pandas as pd

PathLike = Union[str, Path]

def save_csv(df: pd.DataFrame, path: PathLike) -> None:
    """
    Save a DataFrame to CSV (no index). Ensures parent dir exists.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

def load_csv(path: PathLike) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.
    """
    return pd.read_csv(path)
