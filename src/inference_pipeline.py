"""
End-to-end Silambu inference pipeline.

Loads the existing trained artifacts and runs one 30-reading sensor window through:
    sensor window
        -> Activity Recognition
        -> Behavior Learning
        -> Isolation Forest anomaly scoring
        -> XGBoost risk prediction
        -> LOW / MEDIUM / HIGH / CRITICAL

Important:
- This file does NOT retrain any model.
- It uses the existing artifacts under models/.
- Feature schemas are read from the saved scalers/models where possible.
- The anomaly stage uses the same 26 temporal window features used during
  Isolation Forest training.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "silambu_sensor_data.csv"

ACTIVITY_MODEL_PATH = PROJECT_ROOT / "models" / "activity_model.keras"
ACTIVITY_ENCODER_PATH = PROJECT_ROOT / "models" / "activity_label_encoder.pkl"
ACTIVITY_SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

BEHAVIOR_MODEL_PATH = PROJECT_ROOT / "models" / "behavior_model.keras"
BEHAVIOR_SCALER_PATH = PROJECT_ROOT / "models" / "behavior_scaler.pkl"

ANOMALY_MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_detector.pkl"
ANOMALY_SCALER_PATH = PROJECT_ROOT / "models" / "anomaly_scaler.pkl"
ANOMALY_METADATA_PATH = PROJECT_ROOT / "models" / "anomaly_metadata.json"

RISK_MODEL_PATH = PROJECT_ROOT / "models" / "risk_model.pkl"
RISK_METADATA_PATH = PROJECT_ROOT / "models" / "risk_metadata.json"


WINDOW_LENGTH = 30

RISK_CLASS_NAMES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

RISK_FEATURE_NAMES = [
    "heart_rate",
    "spo2",
    "acceleration_magnitude",
    "gyroscope_magnitude",
    "speed",
    "heart_rate_change",
    "speed_change",
    "distance_from_normal_baseline",
    "maximum_distance_from_normal_baseline",
    "activity_encoded",
    "isolation_forest_anomaly_score",
    "tamper",
    "sos",
    "hour_of_day",
]

ANOMALY_FEATURE_NAMES = [
    "mean_acceleration_magnitude",
    "std_acceleration_magnitude",
    "max_acceleration_magnitude",
    "mean_gyroscope_magnitude",
    "std_gyroscope_magnitude",
    "max_gyroscope_magnitude",
    "mean_heart_rate",
    "std_heart_rate",
    "min_heart_rate",
    "max_heart_rate",
    "heart_rate_range",
    "mean_absolute_heart_rate_change",
    "mean_speed",
    "std_speed",
    "max_speed",
    "speed_range",
    "mean_absolute_speed_change",
    "latitude_displacement",
    "longitude_displacement",
    "total_approximate_displacement",
    "distance_from_normal_baseline",
    "maximum_distance_from_normal_baseline",
    "mean_spo2",
    "min_spo2",
    "tamper_count",
    "sos_count",
]


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    with path.open("rb") as file:
        return pickle.load(file)


def require_columns(df: pd.DataFrame, columns: list[str], stage: str)-> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{stage}: missing required sensor columns: {missing}"
        )


def prepare_sensor_window(
    window: pd.DataFrame,
    start_index: int,
) -> pd.DataFrame:
    """Validate and chronologically prepare exactly 30 sensor readings."""
    if len(window) != WINDOW_LENGTH:
        raise ValueError(
            f"Expected exactly {WINDOW_LENGTH} readings, got {len(window)}."
        )

    required = [
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
    ]
    require_columns(window, required, "Input window")

    window = window.copy()

    window["timestamp"] = pd.to_datetime(
        window["timestamp"],
        errors="coerce",
    )

    if window["timestamp"].isna().any():
        raise ValueError("Input window contains invalid timestamps.")

    window = (
        window.sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )

    print(f"Using readings {start_index} to {start_index + WINDOW_LENGTH - 1}")
    print(
        f"Window time: {window['timestamp'].iloc[0]} "
        f"-> {window['timestamp'].iloc[-1]}"
    )

    return window


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the common derived sensor features used by the models."""
    df = df.copy()

    df["acceleration_magnitude"] = np.sqrt(
        df["accel_x"] ** 2
        + df["accel_y"] ** 2
        + df["accel_z"] ** 2
    )

    df["gyroscope_magnitude"] = np.sqrt(
        df["gyro_x"] ** 2
        + df["gyro_y"] ** 2
        + df["gyro_z"] ** 2
    )

    df["heart_rate_change"] = (
        df["heart_rate"].diff().fillna(0.0).abs()
    )

    df["speed_change"] = (
        df["speed"].diff().fillna(0.0).abs()
    )

    return df


def get_feature_names_from_scaler(
    scaler,
    fallback: list[str],
) -> list[str]:
    names = getattr(scaler, "feature_names_in_", None)

    if names is not None:
        return [str(name) for name in names]

    return fallback


def resolve_feature_series(
    df: pd.DataFrame,
    name: str,
    baseline_latitude: float | None = None,
    baseline_longitude: float | None = None,
) -> pd.Series:
    """
    Resolve a saved scaler feature name from raw/derived sensor data.

    This deliberately fails instead of silently inventing a feature if the
    existing artifact expects a feature that this pipeline cannot reproduce.
    """
    if name in df.columns:
        return df[name]

    if name == "latitude_delta":
        return (df["latitude"] - float(baseline_latitude)).abs()

    if name == "longitude_delta":
        return (df["longitude"] - float(baseline_longitude)).abs()

    if name == "distance_from_baseline_location":
        return np.hypot(
            df["latitude"] - float(baseline_latitude),
            df["longitude"] - float(baseline_longitude),
        )

    if name == "distance_from_normal_baseline":
        return np.hypot(
            df["latitude"] - float(baseline_latitude),
            df["longitude"] - float(baseline_longitude),
        )

    if name == "maximum_distance_from_normal_baseline":
        distance = np.hypot(
            df["latitude"] - float(baseline_latitude),
            df["longitude"] - float(baseline_longitude),
        )
        return distance.cummax()

    aliases = {
        "accel_magnitude": "acceleration_magnitude",
        "gyro_magnitude": "gyroscope_magnitude",
        "hr_change": "heart_rate_change",
        "velocity": "speed",
    }

    if name in aliases and aliases[name] in df.columns:
        return df[aliases[name]]

    raise ValueError(
        f"Cannot construct feature '{name}' from the current sensor schema."
    )


def build_sequence_features(
    window: pd.DataFrame,
    scaler,
    stage: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the exact feature order expected by an existing sequence scaler.

    The scaler determines the feature order. This avoids hard-coding a new
    feature order that could disagree with the trained artifact.
    """
    df = add_derived_features(window)

    feature_names = get_feature_names_from_scaler(
        scaler,
        [
            "heart_rate",
            "spo2",
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "speed",
        ],
    )

    values = []
    for name in feature_names:
        values.append(
            np.asarray(
                resolve_feature_series(df, name),
                dtype=float,
            )
        )

    matrix = np.column_stack(values)

    expected_count = getattr(scaler, "n_features_in_", matrix.shape[1])
    if matrix.shape[1] != expected_count:
        raise ValueError(
            f"{stage}: scaler expects {expected_count} features, "
            f"but pipeline created {matrix.shape[1]}."
        )

    scaled = scaler.transform(
        pd.DataFrame(matrix, columns=feature_names)
    )

    return scaled.astype(np.float32), feature_names


def predict_activity(window: pd.DataFrame) -> dict[str, object]:
    """Run the existing DeepConvLSTM activity model."""
    if not ACTIVITY_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing artifact: {ACTIVITY_MODEL_PATH}")

    model = tf.keras.models.load_model(ACTIVITY_MODEL_PATH)
    encoder = load_pickle(ACTIVITY_ENCODER_PATH)

    # Current Activity model contract: 30 readings x 11 features.
    if len(model.input_shape) != 3:
        raise ValueError(
            f"Activity model must have a 3D input shape, got {model.input_shape}."
        )
    if int(model.input_shape[1]) != WINDOW_LENGTH:
        raise ValueError(
            f"Activity model expects {model.input_shape[1]} timesteps, "
            f"but pipeline uses {WINDOW_LENGTH}."
        )
    if int(model.input_shape[2]) != 11:
        raise ValueError(
            f"Activity model must use 11 features, got {model.input_shape[2]}."
        )

    scaler = None
    if ACTIVITY_SCALER_PATH.exists():
        scaler = load_pickle(ACTIVITY_SCALER_PATH)

    if scaler is not None:
        sequence, feature_names = build_sequence_features(
            window,
            scaler,
            "Activity",
        )
    else:
        # The trained model itself still tells us the required feature count.
        feature_count = int(model.input_shape[-1])

        df = add_derived_features(window)

        fallback_features = [
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

        if feature_count != len(fallback_features):
            raise ValueError(
                f"Activity scaler is missing and the model expects "
                f"{feature_count} features, but the known Activity schema "
                f"contains {len(fallback_features)} features."
            )

        sequence = df[fallback_features].to_numpy(dtype=np.float32)
        feature_names = fallback_features

    x = np.expand_dims(sequence, axis=0)

    if x.shape[1] != WINDOW_LENGTH:
        raise ValueError(
            f"Activity expects a {WINDOW_LENGTH}-reading window, "
            f"got {x.shape[1]}."
        )

    if x.shape[2] != int(model.input_shape[-1]):
        raise ValueError(
            f"Activity expects {model.input_shape[-1]} features, "
            f"got {x.shape[2]}."
        )

    probabilities = model.predict(x, verbose=0)[0]
    class_index = int(np.argmax(probabilities))

    try:
        class_name = str(
            encoder.inverse_transform([class_index])[0]
        )
    except Exception:
        class_name = str(class_index)

    return {
        "class_index": class_index,
        "class_name": class_name,
        "confidence": float(probabilities[class_index]),
        "probabilities": probabilities.astype(float).tolist(),
        "feature_names": feature_names,
    }


def predict_behavior(window: pd.DataFrame) -> dict[str, object]:
    """Run the existing Behavior LSTM model."""
    if not BEHAVIOR_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing artifact: {BEHAVIOR_MODEL_PATH}")

    model = tf.keras.models.load_model(BEHAVIOR_MODEL_PATH)
    scaler = load_pickle(BEHAVIOR_SCALER_PATH)

    sequence, feature_names = build_sequence_features(
        window,
        scaler,
        "Behavior",
    )

    x = np.expand_dims(sequence, axis=0)

    if x.shape[1] != WINDOW_LENGTH:
        raise ValueError(
            f"Behavior expects a {WINDOW_LENGTH}-reading window, "
            f"got {x.shape[1]}."
        )

    if x.shape[2] != int(model.input_shape[-1]):
        raise ValueError(
            f"Behavior expects {model.input_shape[-1]} features, "
            f"got {x.shape[2]}."
        )

    output = model.predict(x, verbose=0)
    probabilities = np.asarray(output[0], dtype=float)

    class_index = int(np.argmax(probabilities))

    # Behavior class labels are not known from the artifact names alone.
    # If a behavior label encoder exists in the future, it can be added here.
    class_name = f"class_{class_index}"

    return {
        "class_index": class_index,
        "class_name": class_name,
        "confidence": float(probabilities[class_index]),
        "probabilities": probabilities.tolist(),
        "feature_names": feature_names,
    }


def build_anomaly_window_features(
    window: pd.DataFrame,
    baseline_latitude: float,
    baseline_longitude: float,
) -> pd.DataFrame:
    """
    Reproduce the exact 26 temporal features used by anomaly_detector.py.
    """
    df = add_derived_features(window)

    latitude_delta = (
        df["latitude"] - baseline_latitude
    ).abs()

    longitude_delta = (
        df["longitude"] - baseline_longitude
    ).abs()

    distance = np.hypot(
        df["latitude"] - baseline_latitude,
        df["longitude"] - baseline_longitude,
    )

    acceleration = df["acceleration_magnitude"]
    gyroscope = df["gyroscope_magnitude"]

    feature_row = {
        "mean_acceleration_magnitude": float(acceleration.mean()),
        "std_acceleration_magnitude": float(acceleration.std(ddof=0)),
        "max_acceleration_magnitude": float(acceleration.max()),
        "mean_gyroscope_magnitude": float(gyroscope.mean()),
        "std_gyroscope_magnitude": float(gyroscope.std(ddof=0)),
        "max_gyroscope_magnitude": float(gyroscope.max()),
        "mean_heart_rate": float(df["heart_rate"].mean()),
        "std_heart_rate": float(df["heart_rate"].std(ddof=0)),
        "min_heart_rate": float(df["heart_rate"].min()),
        "max_heart_rate": float(df["heart_rate"].max()),
        "heart_rate_range": float(
            df["heart_rate"].max() - df["heart_rate"].min()
        ),
        "mean_absolute_heart_rate_change": float(
            df["heart_rate_change"].mean()
        ),
        "mean_speed": float(df["speed"].mean()),
        "std_speed": float(df["speed"].std(ddof=0)),
        "max_speed": float(df["speed"].max()),
        "speed_range": float(
            df["speed"].max() - df["speed"].min()
        ),
        "mean_absolute_speed_change": float(
            df["speed_change"].mean()
        ),
        "latitude_displacement": float(latitude_delta.max()),
        "longitude_displacement": float(longitude_delta.max()),
        "total_approximate_displacement": float(
            np.hypot(
                latitude_delta.max(),
                longitude_delta.max(),
            )
        ),
        "distance_from_normal_baseline": float(distance.iloc[-1]),
        "maximum_distance_from_normal_baseline": float(distance.max()),
        "mean_spo2": float(df["spo2"].mean()),
        "min_spo2": float(df["spo2"].min()),
        "tamper_count": int(df["tamper"].sum()),
        "sos_count": int(df["sos"].sum()),
    }

    return pd.DataFrame(
        [feature_row],
        columns=ANOMALY_FEATURE_NAMES,
    )


def predict_anomaly(
    window: pd.DataFrame,
) -> dict[str, object]:
    """Run the existing Isolation Forest on one 30-reading window."""
    model = load_pickle(ANOMALY_MODEL_PATH)
    scaler = load_pickle(ANOMALY_SCALER_PATH)

    if ANOMALY_METADATA_PATH.exists():
        with ANOMALY_METADATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)
    else:
        metadata = {}

    metadata_window_length = int(
        metadata.get("window_length", WINDOW_LENGTH)
    )

    if metadata_window_length != WINDOW_LENGTH:
        raise ValueError(
            f"Saved anomaly detector expects {metadata_window_length}readings, "
            f"but this pipeline is configured for {WINDOW_LENGTH}."
        )

    baseline_latitude = float(
        metadata["normal_baseline_latitude"]
    )
    baseline_longitude = float(
        metadata["normal_baseline_longitude"]
    )

    feature_frame = build_anomaly_window_features(
        window,
        baseline_latitude,
        baseline_longitude,
    )

    expected_columns = list(
        getattr(
            scaler,
            "feature_names_in_",
            ANOMALY_FEATURE_NAMES,
        )
    )

    feature_frame = feature_frame[
        expected_columns
    ]

    scaled_array = scaler.transform(feature_frame)

    # IsolationForest was fitted with named columns. Pass the scaled
    # values back as a DataFrame with the same names to avoid sklearn's
    # "X does not have valid feature names" warning.
    scaled_frame = pd.DataFrame(
        scaled_array,
        columns=expected_columns,
        index=feature_frame.index,
    )

    decision_score = float(
        model.decision_function(scaled_frame)[0]
    )

    prediction = int(
        model.predict(scaled_frame)[0]
    )

    is_anomaly = prediction == -1

    return {
        "decision_score": decision_score,
        "prediction": prediction,
        "is_anomaly": is_anomaly,
        "baseline_latitude": baseline_latitude,
        "baseline_longitude": baseline_longitude,
        "feature_names": expected_columns,
    }


def build_risk_features(
    window: pd.DataFrame,
    activity_class_index: int,
    anomaly_score: float,
) -> pd.DataFrame:
    """
    Build the 14 features expected by the existing Risk XGBoost model.

    This mirrors the feature names in risk_model.py. The Activity prediction
    is used for activity_encoded instead of reading the dataset's activity
    label, which makes this an actual model-to-model connection.
    """
    with RISK_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    baseline_latitude = float(
        metadata["baseline_latitude"]
    )
    baseline_longitude = float(
        metadata["baseline_longitude"]
    )

    df = add_derived_features(window)

    distance = np.hypot(
        df["latitude"] - baseline_latitude,
        df["longitude"] - baseline_longitude,
    )

    # The existing risk training code uses row-level values for these
    # features and cumulative maximum for maximum distance.
    feature_df = pd.DataFrame(
        {
            "heart_rate": df["heart_rate"].to_numpy(),
            "spo2": df["spo2"].to_numpy(),
            "acceleration_magnitude": df[
                "acceleration_magnitude"
            ].to_numpy(),
            "gyroscope_magnitude": df[
                "gyroscope_magnitude"
            ].to_numpy(),
            "speed": df["speed"].to_numpy(),
            "heart_rate_change": df[
                "heart_rate_change"
            ].to_numpy(),
            "speed_change": df[
                "speed_change"
            ].to_numpy(),
            "distance_from_normal_baseline": distance.to_numpy(),
            "maximum_distance_from_normal_baseline": distance.cummax().to_numpy(),
            "activity_encoded": np.full(
                len(df),
                int(activity_class_index),
                dtype=int,
            ),
            "isolation_forest_anomaly_score": np.full(
                len(df),
                float(anomaly_score),
            ),
            "tamper": df["tamper"].astype(int).to_numpy(),
            "sos": df["sos"].astype(int).to_numpy(),
            "hour_of_day": pd.to_datetime(
                df["timestamp"]
            ).dt.hour.to_numpy(),
        }
    )

    feature_df = feature_df[RISK_FEATURE_NAMES]

    return feature_df


def predict_risk(
    window: pd.DataFrame,
    activity_class_index: int,
    anomaly_score: float,
) -> dict[str, object]:
    """Run the existing XGBoost risk model."""
    model = load_pickle(RISK_MODEL_PATH)

    feature_frame = build_risk_features(
        window,
        activity_class_index,
        anomaly_score,
    )

    # Risk model was trained on row-level samples. For a 30-reading window,
    # obtain one prediction per row and aggregate by majority vote.
    predictions = model.predict(feature_frame).astype(int)

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(
            model.predict_proba(feature_frame),
            dtype=float,
        )

    counts = np.bincount(
        predictions,
        minlength=len(RISK_CLASS_NAMES),
    )

    final_class_index = int(np.argmax(counts))
    final_class_name = RISK_CLASS_NAMES[
        final_class_index
    ]

    result = {
        "class_index": final_class_index,
        "risk_level": final_class_name,
        "row_predictions": predictions.tolist(),
        "vote_counts": {
            RISK_CLASS_NAMES[index]: int(counts[index])
            for index in range(len(RISK_CLASS_NAMES))
        },
        "feature_names": RISK_FEATURE_NAMES,
    }

    if probabilities is not None:
        mean_probabilities = probabilities.mean(axis=0)
        result["mean_probabilities"] = (
            mean_probabilities.tolist()
        )
        result["confidence"] = float(
            mean_probabilities[final_class_index]
        )

    return result


def run_pipeline(
    window: pd.DataFrame,
    start_index: int = 0,
) -> dict[str, object]:
    """Run the complete four-stage inference pipeline."""
    window = prepare_sensor_window(
        window,
        start_index,
    )

    print("\n[1/4] Activity Recognition")
    activity = predict_activity(window)
    print(
        f"Activity: {activity['class_name']} "
        f"({activity['confidence']:.4f})"
    )

    print("\n[2/4] Behavior Learning")
    behavior = predict_behavior(window)
    print(
        f"Behavior: {behavior['class_name']} "
        f"({behavior['confidence']:.4f})"
    )

    print("\n[3/4] Anomaly Detection")
    anomaly = predict_anomaly(window)
    print(
        f"Anomaly score: {anomaly['decision_score']:.6f}"
    )
    print(
        f"Anomaly detected: {anomaly['is_anomaly']}"
    )

    print("\n[4/4] Risk Prediction")
    risk = predict_risk(
        window,
        activity["class_index"],
        anomaly["decision_score"],
    )
    print(
        f"Risk: {risk['risk_level']}"
    )

    return {
        "window": {
            "start_index": start_index,
            "length": len(window),
            "start_timestamp": str(window["timestamp"].iloc[0]),
            "end_timestamp": str(window["timestamp"].iloc[-1]),
        },
        "activity": activity,
        "behavior": behavior,
        "anomaly": anomaly,
        "risk": risk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete Silambu AI inference pipeline."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting row in the raw dataset (default: 0).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(RAW_DATA_PATH),
        help="CSV containing sensor readings.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    if args.start < 0:
        raise ValueError("--start must be >= 0.")

    end = args.start + WINDOW_LENGTH

    if end > len(df):
        raise ValueError(
            f"Need {WINDOW_LENGTH} readings starting at {args.start},"
            f"but dataset has only {len(df)} rows."
        )

    window = df.iloc[
        args.start:end
    ].copy()

    results = run_pipeline(
        window,
        args.start,
    )

    print("\n" + "=" * 60)
    print("FINAL SILAMBU DECISION")
    print("=" * 60)
    print(
        f"Activity : {results['activity']['class_name']}"
    )
    print(
        f"Behavior : {results['behavior']['class_name']}"
    )
    print(
        f"Anomaly  : {results['anomaly']['is_anomaly']} "
        f"(score={results['anomaly']['decision_score']:.6f})"
    )
    print(
        f"Risk     : {results['risk']['risk_level']}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()