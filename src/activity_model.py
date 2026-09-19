"""Activity modeling for the Silambu child safety wearable prototype.

Prepares sliding-window sensor data and trains a DeepConvLSTM activity
recognition model using exactly 11 activity features.
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
except ImportError:
    tf = None
    keras = None
    layers = None

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "silambu_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

WINDOW_SIZE = 30

# IMPORTANT:
# The Activity model now uses exactly the same 11-feature contract
# represented by models/scaler.pkl.
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
    "acceleration_magnitude",
    "gyroscope_magnitude",
]

TARGET_COLUMN = "activity"
NUM_FEATURES = len(FEATURE_COLUMNS)


# ---------------------------------------------------------------------
# Data loading / validation
# ---------------------------------------------------------------------

def load_processed_dataset(file_path: Path) -> pd.DataFrame:
    """Load the processed sensor dataset."""
    if not file_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    # Create derived features if preprocessing did not already create them.
    if "acceleration_magnitude" not in df.columns:
        df["acceleration_magnitude"] = np.sqrt(
            df["accel_x"] ** 2
            + df["accel_y"] ** 2
            + df["accel_z"] ** 2
        )

    if "gyroscope_magnitude" not in df.columns:
        df["gyroscope_magnitude"] = np.sqrt(
            df["gyro_x"] ** 2
            + df["gyro_y"] ** 2
            + df["gyro_z"] ** 2
        )

    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Ensure all 11 Activity features and the target exist."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing required Activity columns: "
            f"{missing}"
        )


def validate_feature_contract(X: np.ndarray) -> None:
    """Ensure the prepared tensor matches the Activity model contract."""
    expected_shape = (WINDOW_SIZE, NUM_FEATURES)

    if X.ndim != 3:
        raise ValueError(
            f"Activity input must be 3-dimensional, got shape {X.shape}"
        )

    if X.shape[1:] != expected_shape:
        raise ValueError(
            f"Activity feature contract mismatch. "
            f"Expected each window to have shape {expected_shape}, "
            f"but got {X.shape[1:]}"
        )


# ---------------------------------------------------------------------
# Sliding-window preparation
# ---------------------------------------------------------------------

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
        end_index = start_index + window_size

        window = feature_matrix[start_index:end_index]

        # The activity label of the last reading represents the window.
        label = activity_values[end_index - 1]

        sequences.append(window)
        targets.append(label)

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(targets)

    validate_feature_contract(X)

    return X, y


def encode_activity_labels(
    y: np.ndarray,
) -> tuple[np.ndarray, LabelEncoder]:
    """Encode activity labels into integer class IDs."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y)

    return encoded.astype(np.int64), encoder


def split_sequences(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a stratified 80/20 train/test split."""
    if len(np.unique(y)) < 2:
        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


# ---------------------------------------------------------------------
# Saving prepared arrays / encoder
# ---------------------------------------------------------------------

def save_numpy_array(array: np.ndarray, file_path: Path) -> None:
    """Save a NumPy array."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, array)


def save_label_encoder(
    encoder: LabelEncoder,
    file_path: Path,
) -> None:
    """Save the Activity label encoder."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("wb") as encoder_file:
        pickle.dump(encoder, encoder_file)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_model(
    input_shape: tuple[int, int] = (WINDOW_SIZE, NUM_FEATURES),
    num_classes: int = 8,
):
    """Build the 11-feature DeepConvLSTM Activity model."""
    if tf is None or keras is None or layers is None:
        raise ModuleNotFoundError(
            "TensorFlow is required to build and train the DeepConvLSTM model."
        )

    if input_shape != (WINDOW_SIZE, NUM_FEATURES):
        raise ValueError(
            f"Activity model must use input shape "
            f"({WINDOW_SIZE}, {NUM_FEATURES}), got {input_shape}"
        )

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv1D(
                filters=32,
                kernel_size=3,
                padding="same",
                activation="relu",
            ),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),

            layers.Conv1D(
                filters=64,
                kernel_size=3,
                padding="same",
                activation="relu",
            ),
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


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
):
    """Train the model and return the training history."""
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

    return model, history


def save_history(
    history,
    output_path: Path,
) -> None:
    """Save training history as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=2)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, str, np.ndarray]:
    """Evaluate the trained Activity model."""
    validate_feature_contract(X_test)

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0,
    )

    predictions = model.predict(
        X_test,
        verbose=0,
    )

    predicted_labels = predictions.argmax(axis=1)

    unique_labels = np.unique(
        np.concatenate([y_test, predicted_labels])
    )

    report = classification_report(
        y_test,
        predicted_labels,
        labels=unique_labels,
        zero_division=0,
    )

    conf_matrix = confusion_matrix(
        y_test,
        predicted_labels,
        labels=unique_labels,
    )

    return (
        float(test_loss),
        float(test_accuracy),
        report,
        conf_matrix,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Prepare data, train the 11-feature Activity model, and evaluate it."""

    print("=" * 80)
    print("SILAMBU ACTIVITY MODEL")
    print("=" * 80)
    print(f"Window size : {WINDOW_SIZE}")
    print(f"Features    : {NUM_FEATURES}")
    print("Feature list:")
    for index, feature in enumerate(FEATURE_COLUMNS, start=1):
        print(f"  {index:2d}. {feature}")

    # 1. Load data
    df = load_processed_dataset(PROCESSED_DATA_PATH)
    validate_columns(df)

    print(f"\nLoaded dataset: {df.shape}")
    print(f"Using {NUM_FEATURES} Activity features.")

    # 2. Create sliding windows
    X, y_raw = create_time_series_windows(
        df,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        WINDOW_SIZE,
    )

    print(f"\nWindowed data shape: {X.shape}")

    # 3. Encode labels
    y, label_encoder = encode_activity_labels(y_raw)

    print(f"Activity classes: {list(label_encoder.classes_)}")

    # 4. Split
    X_train, X_test, y_train, y_test = split_sequences(X, y)

    print(f"\nX_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test : {y_test.shape}")

    # 5. Save prepared arrays
    save_numpy_array(X_train, OUTPUT_DIR / "X_train.npy")
    save_numpy_array(X_test, OUTPUT_DIR / "X_test.npy")
    save_numpy_array(y_train, OUTPUT_DIR / "y_train.npy")
    save_numpy_array(y_test, OUTPUT_DIR / "y_test.npy")

    # 6. Save label encoder
    save_label_encoder(
        label_encoder,
        MODEL_DIR / "activity_label_encoder.pkl",
    )

    # 7. Build model with 30 x 11 input
    model = build_model(
        input_shape=(WINDOW_SIZE, NUM_FEATURES),
        num_classes=len(label_encoder.classes_),
    )

    print("\nModel input shape:", model.input_shape)
    model.summary()

    # 8. Train
    trained_model, history = train_model(
        model,
        X_train,
        y_train,
    )

    # 9. Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "activity_model.keras"
    trained_model.save(model_path)

    # 10. Save history
    save_history(
        history,
        OUTPUT_DIR / "activity_training_history.json",
    )

    # 11. Evaluate
    test_loss, test_accuracy, report, conf_matrix = evaluate_model(
        trained_model,
        X_test,
        y_test,
    )

    print("\n" + "=" * 80)
    print("FINAL ACTIVITY MODEL RESULTS")
    print("=" * 80)

    print(
        "Training accuracy:",
        history.history["accuracy"][-1],
    )

    print(
        "Validation accuracy:",
        history.history["val_accuracy"][-1],
    )

    print(
        "Test accuracy:",
        test_accuracy,
    )

    print(
        "Test loss:",
        test_loss,
    )

    print("\nClassification report:\n")
    print(report)

    print("Confusion matrix:\n")
    print(conf_matrix)

    print("\nSaved model:")
    print(model_path)

    print("\nActivity model contract:")
    print(f"  Window: {WINDOW_SIZE}")
    print(f"  Features: {NUM_FEATURES}")
    print(f"  Input shape: {trained_model.input_shape}")

    if trained_model.input_shape != (None, WINDOW_SIZE, NUM_FEATURES):
        raise RuntimeError(
            "Saved Activity model does not have the expected "
            f"(None, {WINDOW_SIZE}, {NUM_FEATURES}) input shape."
        )

    print("\nActivity model successfully trained with 11 features.")


if __name__ == "__main__":
    main()
