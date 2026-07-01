import pandas as pd
from src.config import ANOMALY_LABEL, DATA_PATH, NORMAL_LABEL, PROCESSED_DIR
from src.eda import run_eda
from src.preprocess import run_preprocess

def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

def dataset_info(df: pd.DataFrame) -> None:
    print(f"\nShape: {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"\nLabel Distribution:")
    print(f"\nNormal ({NORMAL_LABEL}): {(df['label'] == NORMAL_LABEL).sum()}")
    print(f"\nAnomaly ({ANOMALY_LABEL}): {(df['label'] == ANOMALY_LABEL).sum()}")

def split_normal_anomaly(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_normal = df[df['label'] == NORMAL_LABEL].copy()
    df_anomaly = df[df['label'] == ANOMALY_LABEL].copy()
    return df_normal, df_anomaly

def save_splits(df_normal: pd.DataFrame, df_anomaly: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_normal.to_csv(PROCESSED_DIR / "df_normal.csv", index=False)
    df_anomaly.to_csv(PROCESSED_DIR / "df_anomaly.csv", index=False)
    print(f"\nSaved normal and anomaly splits to /{PROCESSED_DIR.name}/")

def main() -> None:
    df = load_dataset()
    dataset_info(df)

    df_normal, df_anomaly = split_normal_anomaly(df)
    save_splits(df_normal, df_anomaly)

    run_eda(df)
    run_preprocess(df, df_normal)

if __name__ == "__main__":
    main()