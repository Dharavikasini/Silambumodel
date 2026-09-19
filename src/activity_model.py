"""Activity modeling for the Silambu child safety wearable prototype.

This module prepares the sliding-window sensor data and also trains a
DeepConvLSTM activity recognition model from the prepared arrays.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:  # pragma: no cover - TensorFlow is required at runtime.
    tf = None
    keras = None
    layers = None

try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:  # pragma: no cover - metrics fallback for minimal environments
    classification_report = None
    confusion_matrix = None

try:
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - fallback for minimal environments
    def train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
        rng = np.random.RandomState(random_state)
        indices = np.arange(len(X))
        rng.shuffle(indices)
        split_index = int(len(X) * (1 - test_size))
        train_idx = indices[:split_index]
        test_idx = indices[split_index:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

try:
    from sklearn.preprocessing import LabelEncoder
except ImportError:  # pragma: no cover - fallback for minimal environments
    class LabelEncoder:
        """Minimal fallback label encoder used when scikit-learn is unavailable."""

        def __init__(self) -> None:
            self.classes_: np.ndarray | None = None
            self._mapping: dict[object, int] | None = None

        def fit(self, values):
            unique_values = np.unique(np.asarray(list(values)))
            self.classes_ = unique_values
            self._mapping = {value: index for index, value in enumerate(unique_values)}
            return self

        def transform(self, values):
            if self.classes_ is None or self._mapping is None:
                raise ValueError("LabelEncoder has not been fitted yet.")
            return np.asarray([self._mapping.get(value, -1) for value in values], dtype=np.int64)

        def fit_transform(self, values):
            self.fit(values)
            return self.transform(values)

        def inverse_transform(self, values):
            if self.classes_ is None:
                raise ValueError("LabelEncoder has not been fitted yet.")
            values = np.asarray(values)
            return np.asarray([self.classes_[int(v)] for v in values])


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "silambu_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
WINDOW_SIZE = 30
FEATURE_COLUMNS = [
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
TARGET_COLUMN = "activity"


def load_processed_dataset(file_path: Path) -> pd.DataFrame:
    """Load the processed CSV file generated during preprocessing."""
    return pd.read_csv(file_path)


def validate_columns(df: pd.DataFrame) -> None:
    """Ensure the required columns for sequence preparation exist."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for activity preparation: {missing}")


def create_time_series_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences and their activity labels."""
    if len(df) < window_size:
        raise ValueError(
            f"Not enough rows to create windows of length {window_size}. "
            f"Available rows: {len(df)}"
        )

    feature_matrix = df[feature_columns].to_numpy(dtype=np.float32)
    activity_values = df[target_column].to_numpy()

    sequences = []
    targets = []

    for start_index in range(len(df) - window_size + 1):
        window = feature_matrix[start_index : start_index + window_size]
        label = activity_values[start_index + window_size - 1]
        sequences.append(window)
        targets.append(label)

    return np.asarray(sequences, dtype=np.float32), np.asarray(targets)


def encode_activity_labels(y: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    """Encode activity labels into integer classes and return the fitted encoder."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y)
    return encoded.astype(np.int64), encoder


def split_sequences(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the prepared sequences into 80/20 train/test partitions with stratification."""
    if len(np.unique(y)) > 1:
        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

    return train_test_split(X, y, test_size=0.2, random_state=42)


def save_numpy_array(array: np.ndarray, file_path: Path) -> None:
    """Save a NumPy array to disk."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, array)


def save_label_encoder(encoder: LabelEncoder, file_path: Path) -> None:
    """Persist the trained activity label encoder to disk."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as encoder_file:
        pickle.dump(encoder, encoder_file)


def print_activity_summary(
    X: np.ndarray,
    y: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> None:
    """Print a concise summary of the prepared activity recognition dataset."""
    class_distribution = pd.Series(y).value_counts().sort_index()
    print("Activity preparation summary")
    print("=" * 80)
    print(f"Number of sequences: {len(X)}")
    print(f"Number of features: {X.shape[2]}")
    print(f"Sequence length: {X.shape[1]}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Activity classes: {label_encoder.classes_}")
    print("\nClass distribution:")
    print(class_distribution)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the prepared training and test arrays for model training."""
    X_train = np.load(OUTPUT_DIR / "X_train.npy")
    X_test = np.load(OUTPUT_DIR / "X_test.npy")
    y_train = np.load(OUTPUT_DIR / "y_train.npy")
    y_test = np.load(OUTPUT_DIR / "y_test.npy")

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    return X_train, X_test, y_train, y_test


def build_model(input_shape: tuple[int, int] = (30, 9), num_classes: int = 8):
    """Build a DeepConvLSTM model for activity recognition tasks."""
    if tf is None or keras is None or layers is None:
        raise ModuleNotFoundError(
            "TensorFlow is required to build and train the DeepConvLSTM model. "
            "Install TensorFlow in the environment before running this script."
        )

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.25),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """Train the model and return the fitted model, history, and validation split."""
    if len(np.unique(y_train)) > 1:
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train,
        )
    else:
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
        )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train_split,
        y_train_split,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        verbose=1,
        callbacks=[early_stopping],
    )

    return model, history, X_val, y_val


def save_history(history, output_path: Path) -> None:
    """Save the model training history to disk for later visualization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=2)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float, str, np.ndarray]:
    """Evaluate the trained model on the test set and return metrics."""
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test, verbose=0)
    predicted_labels = predictions.argmax(axis=1)

    unique_labels = np.unique(np.concatenate([y_test, predicted_labels]))
    if classification_report is not None:
        report = classification_report(y_test, predicted_labels, labels=unique_labels, zero_division=0)
    else:
        report = "Classification report unavailable because scikit-learn is not installed."

    if confusion_matrix is not None:
        conf_matrix = confusion_matrix(y_test, predicted_labels, labels=unique_labels)
    else:
        conf_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=np.int64)
        for true_label, pred_label in zip(y_test, predicted_labels):
            conf_matrix[unique_labels.tolist().index(true_label), unique_labels.tolist().index(pred_label)] += 1

    return float(test_loss), float(test_accuracy), report, conf_matrix


def main() -> None:
    """Prepare sliding-window data, train the DeepConvLSTM model, and evaluate it."""
    df = load_processed_dataset(PROCESSED_DATA_PATH)
    validate_columns(df)

    X, y_raw = create_time_series_windows(df, FEATURE_COLUMNS, TARGET_COLUMN, WINDOW_SIZE)
    y, label_encoder = encode_activity_labels(y_raw)

    X_train, X_test, y_train, y_test = split_sequences(X, y)

    save_numpy_array(X_train, OUTPUT_DIR / "X_train.npy")
    save_numpy_array(X_test, OUTPUT_DIR / "X_test.npy")
    save_numpy_array(y_train, OUTPUT_DIR / "y_train.npy")
    save_numpy_array(y_test, OUTPUT_DIR / "y_test.npy")
    save_label_encoder(label_encoder, MODEL_DIR / "activity_label_encoder.pkl")

    print_activity_summary(X, y, X_train, X_test, y_train, y_test, label_encoder)

    X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded = load_data()
    model = build_model(input_shape=(30, 9), num_classes=len(label_encoder.classes_))
    model.summary()
    trained_model, history, _, _ = train_model(model, X_train_loaded, y_train_loaded)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trained_model.save(MODEL_DIR / "activity_model.keras")
    save_history(history, OUTPUT_DIR / "activity_training_history.json")

    test_loss, test_accuracy, report, conf_matrix = evaluate_model(trained_model, X_test_loaded, y_test_loaded)

    print("\nTraining accuracy:", history.history["accuracy"][-1])
    print("Validation accuracy:", history.history["val_accuracy"][-1])
    print("Test accuracy:", test_accuracy)
    print("Test loss:", test_loss)
    print("\nClassification report:\n", report)
    print("\nConfusion matrix:\n", conf_matrix)
    print(f"\nSaved trained model to: {MODEL_DIR / 'activity_model.keras'}")
    print(f"Saved training history to: {OUTPUT_DIR / 'activity_training_history.json'}")


if __name__ == "__main__":
    main()
