from __future__ import annotations

import itertools
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.config import (
    MODEL_PATH,
    OCSVM_GAMMA,
    OCSVM_GRID_SEARCH,
    OCSVM_KERNEL,
    OCSVM_NU,
    OCSVM_PARAM_GRID,
    PROCESSED_DIR,
    RANDOM_STATE,
    TRAIN_SCALER_PATH,
)
from src.splits import Splits

@dataclass
class TrainResult:
    model: OneClassSVM
    scaler: StandardScaler
    params: dict
    train_rows: int

def load_splits_from_disk() -> Splits:
    paths = {
        "X_normal_train": PROCESSED_DIR / "X_normal_train.csv",
        "X_normal_test": PROCESSED_DIR / "X_normal_test.csv",
        "X_anomaly_test": PROCESSED_DIR / "X_anomaly_test.csv",
        "X_test": PROCESSED_DIR / "X_test.csv",
        "y_test": PROCESSED_DIR / "y_test.csv",
        "indices": PROCESSED_DIR / "split_indices.pkl",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing split artifact: {path}")

    indices = joblib.load(paths["indices"])
    feature_cols = list(pd.read_csv(paths["X_normal_train"], nrows=0).columns)

    return Splits(
        X_normal_train=pd.read_csv(paths["X_normal_train"]),
        X_normal_test=pd.read_csv(paths["X_normal_test"]),
        X_anomaly_test=pd.read_csv(paths["X_anomaly_test"]),
        X_test=pd.read_csv(paths["X_test"]),
        y_test=pd.read_csv(paths["y_test"]).squeeze("columns"),
        feature_cols=feature_cols,
        train_indices=indices["train_indices"],
        test_indices=indices["test_indices"],
    )

def fit_train_scaler(X_normal_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_normal_train)
    return scaler

def build_ocsvm(
    kernel: str = OCSVM_KERNEL,
    gamma: str | float = OCSVM_GAMMA,
    nu: float = OCSVM_NU,
) -> OneClassSVM:
    return OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)

def _inlier_fraction(model: OneClassSVM, X: np.ndarray) -> float:
    return float((model.predict(X) == 1).mean())

def grid_search_ocsvm(
    X_normal_train_scaled: np.ndarray,
    param_grid: dict | None = None,
    val_size: float = 0.2,
) -> dict:
    grid = param_grid or OCSVM_PARAM_GRID
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    fit_idx, val_idx = train_test_split(
        np.arange(len(X_normal_train_scaled)),
        test_size=val_size,
        random_state=RANDOM_STATE,
    )
    X_fit = X_normal_train_scaled[fit_idx]
    X_val = X_normal_train_scaled[val_idx]

    best_params: dict = {}
    best_score = -1.0

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        model = build_ocsvm(
            kernel=params.get("kernel", OCSVM_KERNEL), 
            gamma=params.get("gamma", OCSVM_GAMMA),
            nu=params.get("nu", OCSVM_NU)
        )
        model.fit(X_fit)
        score = _inlier_fraction(model, X_val)
        if score > best_score:
            best_score = score
            best_params = params

    return best_params

def train_ocsvm(
    splits: Splits,
    use_grid_search: bool = OCSVM_GRID_SEARCH,
) -> TrainResult:
    # Train scaler on normal traffic only
    scaler = fit_train_scaler(splits.X_normal_train)
    X_normal_train_scaled = scaler.transform(splits.X_normal_train)

    params = {
        "kernel": OCSVM_KERNEL,
        "gamma": OCSVM_GAMMA,
        "nu": OCSVM_NU,
    }

    if use_grid_search:
        tuned = grid_search_ocsvm(X_normal_train_scaled)
        params.update(tuned)

    model = build_ocsvm(
        kernel=params["kernel"],
        gamma=params["gamma"],
        nu=params["nu"],
    )
    model.fit(X_normal_train_scaled)

    return TrainResult(
        model=model,
        scaler=scaler,
        params=params,
        train_rows=len(splits.X_normal_train),
    )

def save_model(result: TrainResult) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.model, MODEL_PATH)
    joblib.dump(result.scaler, TRAIN_SCALER_PATH)
    joblib.dump(result.params, PROCESSED_DIR / "ocsvm_params.pkl")

    return {
        "model_path": str(MODEL_PATH),
        "scaler_path": str(TRAIN_SCALER_PATH),
        "params": result.params,
        "train_rows": result.train_rows,
    }

def run_train(splits: Splits | None = None) -> dict:
    if splits is None:
        splits = load_splits_from_disk()

    result = train_ocsvm(splits)
    save_meta = save_model(result)

    return {
        "result": result,
        "save_meta": save_meta,
    }
