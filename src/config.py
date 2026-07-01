from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "embedded_system_network_security_dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data"

NORMAL_LABEL = 0.0
ANOMALY_LABEL = 1.0

TEST_SIZE = 0.2
RANDOM_STATE = 42

OCSVM_KERNEL = "rbf"
OCSVM_GAMMA = "scale"
OCSVM_NU = 0.05

OCSVM_GRID_SEARCH = True
OCSVM_PARAM_GRID = {
    "nu": [0.01, 0.05, 0.1],
    "gamma": ["scale", "auto", 0.1, 1.0],
}

MODEL_PATH = PROCESSED_DIR / "ocsvm_model.pkl"
TRAIN_SCALER_PATH = PROCESSED_DIR / "train_scaler.pkl"
