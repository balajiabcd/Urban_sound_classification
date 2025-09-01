import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

def split_xy(df: pd.DataFrame, target_col: str):
    X = df.iloc[:, 1:-1] if target_col == df.columns[-1] else df.drop(columns=[target_col])
    y = df[target_col]
    return X.select_dtypes(include=[np.number]), y.astype(str)

def make_splits(X, y, test_size=0.2, val_size=0.1, seed=42):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=test_size+val_size, stratify=y, random_state=seed)
    rel = val_size/(test_size+val_size) if (test_size+val_size)>0 else 0.5
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=1-rel, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te

def fit_transform_pca_scaler(X_train, X_val, X_test, n_components=20):
    pca = PCA(n_components=n_components, random_state=0).fit(X_train)
    X_train_p = pca.transform(X_train); X_val_p = pca.transform(X_val); X_test_p = pca.transform(X_test)
    sc = StandardScaler().fit(X_train_p)
    return (sc.transform(X_train_p), sc.transform(X_val_p), sc.transform(X_test_p), pca, sc)

def label_encode(y_tr, y_va, y_te):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y_tr)
    return le.transform(y_tr), le.transform(y_va), le.transform(y_te), le
