from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "embedded_system_network_security_dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data"

NORMAL_LABEL = 0.0
ANOMALY_LABEL = 1.0
