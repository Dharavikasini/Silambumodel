"""Behavior-learning LSTM for Silambu temporal behavior modeling.

This module loads the processed Silambu sensor data, engineers temporal derived
features, creates chronological sequences, and trains an LSTM model to predict the
next behavior/activity state from prior sensor history.

It is intentionally limited to behavior learning and does not implement anomaly
alerting or risk detection.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "silambu_sensor_data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "silambu_processed.csv"
BEHAVIOR_SCALER_PATH = PROJECT_ROOT / "models" / "behavior_scaler.pkl"
LABEL_MAPPING_PATH = PROJECT_ROOT / "data" / "processed" / "behavior_label_mapping.json"
HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "behavior_training_history.json"
MODEL_PATH = PROJECT_ROOT / "models" / "behavior_model.keras"

SEQUENCE_LENGTH = 30
ORIGINAL_SENSOR_COLUMNS = [
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
DERIVED_FEATURE_COLUMNS = [
    "acceleration_magnitude",
    "gyroscope_magnitude",
    "heart_rate_change",
    "speed_change",
]
FINAL_FEATURE_COLUMNS = ORIGINAL_SENSOR_COLUMNS + DERIVED_FEATURE_COLUMNS
ACTIVITY_TO_ID = {
    "resting": 0,
    "walking": 1,
    "running": 2,
    "sudden_movement": 3,
    "struggle": 4,
    "travel": 5,
    "tampering": 6,
    "emergency": 7,
}


def load_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the Silambu sensor dataset used for behavior modeling."""
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("The behavior dataset is empty.")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal sensor features for behavior learning."""
    df = df.copy()

    df["acceleration_magnitude"] = np.sqrt(
        df["accel_x"] ** 2 + df["accel_y"] ** 2 + df["accel_z"] ** 2
    )
    df["gyroscope_magnitude"] = np.sqrt(
        df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2
    )

    df["heart_rate_change"] = df["heart_rate"].diff().fillna(0.0)
    df["speed_change"] = df["speed"].diff().fillna(0.0)

    return df


def save_behavior_scaler(scaler: StandardScaler, output_path: Path = BEHAVIOR_SCALER_PATH) -> None:
    """Persist the behavior-specific scaler for later reuse."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)


def save_label_mapping(mapping: dict[str, int], output_path: Path = LABEL_MAPPING_PATH) -> None:
    """Save the deterministic behavior label mapping to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=2, sort_keys=True)


def validate_activity_labels(labels: pd.Series, mapping: dict[str, int]) -> np.ndarray:
    """Validate labels and map them to deterministic integer IDs."""
    labels = labels.astype(str)
    missing = sorted({label for label in labels if label not in mapping})
    if missing:
        raise ValueError(f"Missing activity mapping for labels: {missing}")
    return labels.map(mapping).to_numpy(dtype=np.int32)


def build_behavior_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the final 13 features from the training dataframe only."""
    scaler = StandardScaler()
    scaler.fit(train_df[FINAL_FEATURE_COLUMNS].astype(float))
    return scaler


def create_sequences(df: pd.DataFrame, labels: np.ndarray, sequence_length: int = SEQUENCE_LENGTH):
    """Create chronological sequences and next-state labels from the dataset."""
    features = df[FINAL_FEATURE_COLUMNS].astype(float).to_numpy()
    if len(features) < sequence_length + 1:
        raise ValueError(
            "Not enough records to create sequences with the requested sequence length."
        )

    sequences = []
    next_labels = []

    for index in range(len(features) - sequence_length):
        seq = features[index : index + sequence_length]
        next_label = labels[index + sequence_length]
        sequences.append(seq)
        next_labels.append(next_label)

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(next_labels, dtype=np.int32)
    return X, y


def build_model(input_shape: tuple[int, int], num_classes: int | None = None):
    """Build a temporal LSTM model to learn behavior patterns from sensor sequences."""
    if num_classes is None:
        num_classes = 8

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """Train the behavior LSTM with validation monitoring and early stopping."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=64,
        verbose=1,
        callbacks=[early_stopping],
    )
    return history


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the trained model on the chronological test set."""
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    return test_loss, test_accuracy


def save_artifacts(model: keras.Model, history, model_path: Path = MODEL_PATH, history_path: Path = HISTORY_PATH):
    """Persist the trained model and training history."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=2)


def main() -> None:
    """Load data, create temporal sequences, train the LSTM, and save artifacts."""
    df = load_data()
    df = create_features(df)

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    scaler = build_behavior_scaler(train_df)
    train_df[FINAL_FEATURE_COLUMNS] = scaler.transform(train_df[FINAL_FEATURE_COLUMNS].astype(float))
    test_df[FINAL_FEATURE_COLUMNS] = scaler.transform(test_df[FINAL_FEATURE_COLUMNS].astype(float))

    save_behavior_scaler(scaler)

    validation_split = int(len(train_df) * 0.2)
    train_df, val_df = train_df[:-validation_split].copy(), train_df[-validation_split:].copy()

    y_train = validate_activity_labels(train_df["activity"].astype(str), ACTIVITY_TO_ID)
    y_val = validate_activity_labels(val_df["activity"].astype(str), ACTIVITY_TO_ID)
    y_test = validate_activity_labels(test_df["activity"].astype(str), ACTIVITY_TO_ID)

    save_label_mapping(ACTIVITY_TO_ID)

    if len(np.unique(y_train)) == 0 or len(np.unique(y_val)) == 0 or len(np.unique(y_test)) == 0:
        raise ValueError("No labels were generated for the behavior training sequences.")

    X_train, y_train = create_sequences(train_df, y_train, SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(val_df, y_val, SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(test_df, y_test, SEQUENCE_LENGTH)

    model = build_model(input_shape=(SEQUENCE_LENGTH, X_train.shape[2]), num_classes=8)
    print(f"Train sequence shape: {X_train.shape}")
    print(f"Validation sequence shape: {X_val.shape}")
    print(f"Test sequence shape: {X_test.shape}")
    print(f"Train labels shape: {y_train.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print("\nModel summary:")
    model.summary()

    history = train_model(model, X_train, y_train, X_val, y_val)

    test_loss, test_accuracy = evaluate_model(model, X_test, y_test)
    print(f"\nTraining accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    save_artifacts(model, history)
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved history to: {HISTORY_PATH}")
    print(f"Saved behavior scaler to: {BEHAVIOR_SCALER_PATH}")
    print(f"Saved label mapping to: {LABEL_MAPPING_PATH}")


if __name__ == "__main__":
    main()
