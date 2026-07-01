import pandas as pd

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

def run_preprocess(df: pd.DataFrame) -> dict:
    X, y, feature_cols, prep_meta = prepare_features(df)

    return {
       "X": X,
        "y": y,
        "feature_cols": feature_cols,
        "prep_meta": prep_meta,
    }