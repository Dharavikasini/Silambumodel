"""Preprocessing pipeline for the Silambu synthetic sensor dataset.

This module loads the raw CSV, validates required fields, cleans and encodes the
sensor data, engineers useful features, scales numeric sensor inputs, and saves
the processed dataset plus the fitted scaler.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "silambu_sensor_data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "silambu_processed.csv"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

REQUIRED_COLUMNS = [
    "timestamp",
    "heart_rate",
    "spo2",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "latitude",
    "longitude",
    "speed",
    "activity",
    "tamper",
    "sos",
    "scenario",
]

NUMERICAL_SENSOR_COLUMNS = [
    "heart_rate",
    "spo2",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "speed",
]

SCALE_COLUMNS = [
    "heart_rate",
    "spo2",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "speed",
    "acceleration_magnitude",
    "gyroscope_magnitude",
]


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the raw CSV dataset from disk."""
    return pd.read_csv(file_path)


def validate_columns(df: pd.DataFrame) -> None:
    """Ensure all required columns exist in the dataset."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def convert_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the timestamp column to pandas datetime format."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def report_dataset_summary(df: pd.DataFrame) -> dict[str, object]:
    """Generate a concise summary of missing values, duplicates, and statistics."""
    missing_values = df.isnull().sum()
    duplicate_rows = df.duplicated().sum()
    numeric_stats = df[NUMERICAL_SENSOR_COLUMNS].describe().transpose()

    print("Preprocessing summary")
    print("=" * 80)
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nMissing values:")
    print(missing_values)
    print(f"\nDuplicate rows: {duplicate_rows}")
    print("\nNumerical sensor statistics:")
    print(numeric_stats)

    return {
        "rows": len(df),
        "missing_values": missing_values.to_dict(),
        "duplicate_rows": duplicate_rows,
        "numeric_stats": numeric_stats,
    }


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create sensor magnitude features from accelerometer and gyroscope readings."""
    df = df.copy()
    df["acceleration_magnitude"] = np.sqrt(
        df["accel_x"] ** 2 + df["accel_y"] ** 2 + df["accel_z"] ** 2
    )
    df["gyroscope_magnitude"] = np.sqrt(
        df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2
    )
    return df


def encode_activity_column(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the categorical activity column using LabelEncoder."""
    df = df.copy()
    label_encoder = LabelEncoder()
    df["activity"] = label_encoder.fit_transform(df["activity"].astype(str))
    return df


def normalize_sensor_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize numeric sensor features while preserving non-normalized fields."""
    df = df.copy()

    scaler = StandardScaler()
    df[SCALE_COLUMNS] = scaler.fit_transform(df[SCALE_COLUMNS])
    return df, scaler


def save_processed_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Persist the processed dataframe to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_scaler(scaler: StandardScaler, output_path: Path) -> None:
    """Persist the fitted StandardScaler object for later reuse."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)


def main() -> None:
    """Execute the preprocessing pipeline for the Silambu synthetic sensor data."""
    df = load_dataset(RAW_DATA_PATH)
    validate_columns(df)
    df = convert_timestamp(df)

    summary = report_dataset_summary(df)
    print(f"\nSummary object: {summary['rows']} rows processed")

    df = add_derived_features(df)
    df = encode_activity_column(df)
    df, scaler = normalize_sensor_features(df)

    save_processed_dataset(df, PROCESSED_DATA_PATH)
    save_scaler(scaler, SCALER_PATH)

    print("\nProcessed dataset saved to:", PROCESSED_DATA_PATH)
    print("Scaler saved to:", SCALER_PATH)
    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
