from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_train_valid_stratified(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/validation split.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def fit_transformers(X: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit preprocessing transformers on X and return them.
    Currently includes only StandardScaler to keep the API simple.
    """
    scaler = StandardScaler().fit(X)
    return {"scaler": scaler}

def apply_transformers(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Apply fitted transformers to a DataFrame and return a transformed DataFrame
    with the same columns and index.
    """
    X2 = scaler.transform(X)
    return pd.DataFrame(X2, columns=X.columns, index=X.index)
