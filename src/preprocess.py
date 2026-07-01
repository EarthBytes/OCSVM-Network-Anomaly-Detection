import joblib
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR

LABEL_COL = "label"
DROP_COLS = ["mean_packet_size"]

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], dict]:
    df = df.copy()  
    meta = {}

    # Missing value imputation
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0]

    imputed = {}
    for col, count in cols_with_missing.items():
        if col == LABEL_COL:
            continue
        median = df[col].median()
        df[col] = df[col].fillna(median)
        imputed[col] = {"count": int(count), "median": float(median)}

    meta["imputed"] = imputed

    # Drop constant features and configured drop list 
    constant_cols = [col for col in df.columns if col != LABEL_COL and df[col].nunique() <= 1]
    drop = list(set(DROP_COLS + constant_cols))

    if drop:
        df = df.drop(columns=drop)
    
    meta["dropped_features"] = drop

    # Encode boolean columns
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    meta["encoded_booleans"] = list(bool_cols)

    # Final feature set
    feature_cols = [col for col in df.columns if col != LABEL_COL]
    X = df[feature_cols].astype(float)
    y = df[LABEL_COL].astype(float)

    meta["feature_count"] = len(feature_cols)

    return X, y, feature_cols, meta

def fit_scaler(X_normal: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_normal)
    return scaler

def scale_feature(X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)

def save_preprossed(
        X: pd.DataFrame,
        y: pd.Series,
        scaler: StandardScaler,
        feature_cols: list[str],
) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_scaled = scale_feature(X, scaler)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    paths = {
        "X_featres": PROCESSED_DIR / "X_features.csv",
        "X_scaled": PROCESSED_DIR / "X_scaled.csv",
        "y_labels": PROCESSED_DIR / "y_labels.csv",
        "scaler": PROCESSED_DIR / "scaler.pkl",
    }      

    X.to_csv(paths["X_featres"], index=False)
    X_scaled_df.to_csv(paths["X_scaled"], index=False)
    y.to_csv(paths["y_labels"], index=False)
    joblib.dump(scaler, paths["scaler"])

    return {
        "saved_paths": {k: str(v) for k, v in paths.items()},
        "rows": len(X),
        "features": len(feature_cols),
    }

def run_preprocess(
        df: pd.DataFrame,
        df_normal: pd.DataFrame
) -> dict:
    X, y, feature_cols, prep_meta = prepare_features(df)
    X_normal, _, _, prep_meta_normal = prepare_features(df_normal)

    scaler = fit_scaler(X_normal)
    save_meta = save_preprossed(X, y, scaler, feature_cols)

    return {
       "X": X,
        "y": y,
        "feature_cols": feature_cols,
        "prep_meta": prep_meta,
        "prep_meta_normal": prep_meta_normal,
        "save_meta": save_meta,
    }