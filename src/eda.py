import pandas as pd
from src.config import ANOMALY_LABEL, NORMAL_LABEL

NUMERIC_FEATURES = [
    "packet_size",
    "inter_arrival_time",
    "src_port",
    "dst_port",
    "packet_count_5s",
    "spectral_entropy",
    "frequency_band_energy",
]

CATEGORICAL_FEATURES = [
    "protocol_type_TCP",
    "protocol_type_UDP",
    "src_ip_192.168.1.2",
    "src_ip_192.168.1.3",
    "dst_ip_192.168.1.5",
    "dst_ip_192.168.1.6",
    "tcp_flags_FIN",
    "tcp_flags_SYN",
    "tcp_flags_SYN-ACK",
]

def class_balance(df: pd.DataFrame) -> dict:
    normal_count = (df['label'] == NORMAL_LABEL).sum()
    anomaly_count = (df['label'] == ANOMALY_LABEL).sum()
    total = len(df)

    return {
        "normal": normal_count,
        "anomaly": anomaly_count,
        "normal_ratio": normal_count / total if total > 0 else 0.0,
    }

def data_leakage_report(df: pd.DataFrame) -> dict:
    feature_cols = [c for c in df.columns if c != 'label']

    constant_features = [col for col in feature_cols if df[col].nunique() <= 1]

    suspicious_label_corr = [
        col 
        for col in feature_cols
        if abs(df[col].astype(float).corr(df["label"])) > 0.95
    ]

    return {
        "label_excluded": "label" not in feature_cols,
        "constant_features": constant_features,
        "high_level_correlation": suspicious_label_corr,
    }

def summarise_categoricals(df: pd.DataFrame) -> dict:
    summary = {}
    
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            true_count = df[col].sum()
            summary[col] = {
                "true_count": int(true_count),
                "true_ratio": float(true_count / len(df))
            }
    
    return summary

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != 'label']
    return df[feature_cols].astype(float).corr()

def run_eda(df: pd.DataFrame) -> dict:
    return {
        "class_balance": class_balance(df),
        "data_leakage": data_leakage_report(df),
        "categorical_summary": summarise_categoricals(df),
        "correlation_matrix": correlation_matrix(df),
    }