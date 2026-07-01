from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import (
    ANOMALY_LABEL,
    NORMAL_LABEL,
    PROCESSED_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.preprocess import prepare_features

@dataclass
class Splits:
    X_normal_train: pd.DataFrame
    X_normal_test: pd.DataFrame
    X_anomaly_test: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_cols: list[str]
    train_indices: np.ndarray
    test_indices: np.ndarray

@dataclass
class SplitMeta:
    train: int
    normal_test: int
    anomaly_test: int
    test: int

def _subset(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    return df.iloc[idx].reset_index(drop=True)

def create_train_test_splits(
    df_normal: pd.DataFrame,
    df_anomaly: pd.DataFrame,
) -> Splits:
    
    # Preprocess both datasets
    X_normal, _, feature_cols, _ = prepare_features(df_normal)
    X_anomaly, _, _, _ = prepare_features(df_anomaly)

    # Train/test split on normal only
    train_idx, normal_test_idx = train_test_split(
        np.arange(len(X_normal)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Map back to original indices
    normal_indices = df_normal.index.to_numpy()
    train_indices = normal_indices[train_idx]
    normal_test_indices = normal_indices[normal_test_idx]
    anomaly_test_indices = df_anomaly.index.to_numpy()

    # Subsets
    X_normal_train = _subset(X_normal, train_idx)
    X_normal_test = _subset(X_normal, normal_test_idx)
    X_anomaly_test = X_anomaly.reset_index(drop=True)

    # Combined test set
    X_test = pd.concat([X_normal_test, X_anomaly_test], ignore_index=True)
    y_test = pd.Series(
        [NORMAL_LABEL] * len(X_normal_test) + [ANOMALY_LABEL] * len(X_anomaly_test),
        name="label",
    )

    # Combined test indices
    test_indices = np.concatenate([normal_test_indices, anomaly_test_indices])

    return Splits(
        X_normal_train=X_normal_train,
        X_normal_test=X_normal_test,
        X_anomaly_test=X_anomaly_test,
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols,
        train_indices=train_indices,
        test_indices=test_indices,
    )

def save_splits(splits: Splits) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    path = {
        "X_normal_train": PROCESSED_DIR / "X_normal_train.csv",
        "X_normal_test": PROCESSED_DIR / "X_normal_test.csv",
        "X_anomaly_test": PROCESSED_DIR / "X_anomaly_test.csv",
        "X_test": PROCESSED_DIR / "X_test.csv",
        "y_test": PROCESSED_DIR / "y_test.csv",
        "indices": PROCESSED_DIR / "split_indices.pkl",
    }

    # Save DataFrames
    for key in ["X_normal_train", "X_normal_test", "X_anomaly_test", "X_test", "y_test"]:
        df = getattr(splits, key)
        df.to_csv(path[key], index=False)

    # Save index metadata
    joblib.dump(
        {
            "train_indices": splits.train_indices,
            "test_indices": splits.test_indices,
        },
        path["indices"],
    )

    return {
        "saved_paths": {k: str(v) for k, v in path.items()},
        "train_rows": len(splits.X_normal_train),
        "normal_test_rows": len(splits.X_normal_test),
        "anomaly_test_rows": len(splits.X_anomaly_test),
        "test_rows": len(splits.X_test),
    }

def run_splits(df_normal: pd.DataFrame, df_anomaly: pd.DataFrame) -> dict:
    splits = create_train_test_splits(df_normal, df_anomaly)
    save_meta = save_splits(splits)
    
    return {
        "splits": splits,
        "meta": SplitMeta(
            train=len(splits.X_normal_train),
            normal_test=len(splits.X_normal_test),
            anomaly_test=len(splits.X_anomaly_test),
            test=len(splits.X_test),
    ),
}