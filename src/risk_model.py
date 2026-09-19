"""XGBoost risk model for Silambu.

This prototype predicts a synthetic risk level from sensor, activity, location,
anomaly, tamper, SOS, and temporal context while keeping the scenario label
only for target mapping and never as an input feature.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "silambu_sensor_data.csv"

ANOMALY_MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_detector.pkl"
ANOMALY_SCALER_PATH = PROJECT_ROOT / "models" / "anomaly_scaler.pkl"
ANOMALY_METADATA_PATH = PROJECT_ROOT / "models" / "anomaly_metadata.json"

RISK_MODEL_PATH = PROJECT_ROOT / "models" / "risk_model.pkl"
RISK_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "risk_model_results.json"
RISK_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "risk_features.json"
RISK_METADATA_PATH = PROJECT_ROOT / "models" / "risk_metadata.json"


RISK_CLASS_MAPPING = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}

RISK_CLASS_NAMES = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]


SCENARIO_TO_RISK = {
    "normal_rest": 0,
    "normal_walking": 0,
    "normal_running": 0,
    "unusual_location": 1,
    "sudden_movement": 2,
    "tampering": 2,
    "possible_struggle": 3,
    "emergency_sos": 3,
}


FEATURE_NAMES = [
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


def load_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Silambu sensor dataset."""

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("The raw dataset is empty.")

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create row-level motion and rate features used for risk prediction."""

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


def calculate_location_features(
    df: pd.DataFrame,
    baseline_latitude: float,
    baseline_longitude: float,
) -> pd.DataFrame:
    """Compute baseline-relative location features and cumulative locality drift."""

    df = df.copy()

    df["latitude_delta"] = (
        df["latitude"] - baseline_latitude
    ).abs()

    df["longitude_delta"] = (
        df["longitude"] - baseline_longitude
    ).abs()

    df["distance_from_baseline_location"] = np.hypot(
        df["latitude"] - baseline_latitude,
        df["longitude"] - baseline_longitude,
    )

    df["distance_from_normal_baseline"] = (
        df["distance_from_baseline_location"].copy()
    )

    df["maximum_distance_from_normal_baseline"] = (
        df["distance_from_normal_baseline"].cummax()
    )

    return df


def create_risk_labels(df: pd.DataFrame) -> np.ndarray:
    """Map scenario labels only to risk classes; scenario is never used as a feature."""

    labels = df["scenario"].map(SCENARIO_TO_RISK)

    missing = labels.isna().sum()

    if missing:
        raise ValueError(
            f"Scenario-to-risk mapping missing for {int(missing)} rows."
        )

    return labels.to_numpy(dtype=int)


def create_model_features(
    df: pd.DataFrame,
    anomaly_model,
    anomaly_scaler,
    activity_map: dict[str, int],
    baseline_latitude: float,
    baseline_longitude: float,
) -> pd.DataFrame:
    """Create the XGBoost feature matrix while matching the saved anomaly scaler schema."""

    df = create_features(df).copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["latitude_delta"] = (
        df["latitude"] - baseline_latitude
    ).abs()

    df["longitude_delta"] = (
        df["longitude"] - baseline_longitude
    ).abs()

    df["distance_from_baseline_location"] = np.hypot(
        df["latitude"] - baseline_latitude,
        df["longitude"] - baseline_longitude,
    )

    df["distance_from_normal_baseline"] = (
        df["distance_from_baseline_location"].copy()
    )

    df["maximum_distance_from_normal_baseline"] = (
        df["distance_from_normal_baseline"].cummax()
    )

    # Features expected by the saved anomaly scaler.
    df["mean_acceleration_magnitude"] = (
        df["acceleration_magnitude"]
    )

    df["std_acceleration_magnitude"] = 0.0

    df["max_acceleration_magnitude"] = (
        df["acceleration_magnitude"]
    )

    df["mean_gyroscope_magnitude"] = (
        df["gyroscope_magnitude"]
    )

    df["std_gyroscope_magnitude"] = 0.0

    df["max_gyroscope_magnitude"] = (
        df["gyroscope_magnitude"]
    )

    df["mean_heart_rate"] = df["heart_rate"]

    df["std_heart_rate"] = 0.0

    df["min_heart_rate"] = df["heart_rate"]

    df["max_heart_rate"] = df["heart_rate"]

    df["heart_rate_range"] = 0.0

    df["mean_absolute_heart_rate_change"] = (
        df["heart_rate_change"]
    )

    df["mean_speed"] = df["speed"]

    df["std_speed"] = 0.0

    df["max_speed"] = df["speed"]

    df["speed_range"] = 0.0

    df["mean_absolute_speed_change"] = (
        df["speed_change"]
    )

    df["latitude_displacement"] = df["latitude_delta"]

    df["longitude_displacement"] = df["longitude_delta"]

    df["total_approximate_displacement"] = np.hypot(
        df["latitude_delta"],
        df["longitude_delta"],
    )

    df["mean_spo2"] = df["spo2"]

    df["min_spo2"] = df["spo2"]

    df["tamper_count"] = df["tamper"].astype(int)

    df["sos_count"] = df["sos"].astype(int)

    expected_columns = list(
        anomaly_scaler.feature_names_in_
    )

    print(
        "Anomaly scaler expects:",
        expected_columns,
    )

    print(
        "Provided anomaly features:",
        list(df.columns),
    )

    missing = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing anomaly scaler features: {missing}"
        )

    features = df[expected_columns].copy()

    if list(features.columns) != list(expected_columns):
        raise ValueError(
            "Anomaly feature column order mismatch. "
            f"Expected: {expected_columns}. "
            f"Provided: {list(features.columns)}"
        )

    features = features[expected_columns]

    anomaly_scaled = pd.DataFrame(
    anomaly_scaler.transform(features),
    columns=expected_columns,
    index=features.index,
)

    df["isolation_forest_anomaly_score"] = (
    anomaly_model.decision_function(
        anomaly_scaled
    )
)
    

    df["activity_encoded"] = (
        df["activity"]
        .map(activity_map)
        .astype(int)
    )

    df["hour_of_day"] = (
        df["timestamp"]
        .dt.hour
        .astype(int)
    )

    feature_frame = df[FEATURE_NAMES].copy()

    return feature_frame


def prepare_data() -> tuple[
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    dict[str, object],
]:
    """Create stratified train/test splits using the risk feature set."""

    df = load_data()

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = df.sort_values(
        "timestamp",
        kind="mergesort",
    ).reset_index(drop=True)

    with ANOMALY_MODEL_PATH.open("rb") as anomaly_model_file:
        anomaly_model = pickle.load(
            anomaly_model_file
        )

    with ANOMALY_SCALER_PATH.open("rb") as anomaly_scaler_file:
        anomaly_scaler = pickle.load(
            anomaly_scaler_file
        )

    # ---------------------------------------------------------
    # Create risk labels before splitting.
    # ---------------------------------------------------------

    risk_labels = create_risk_labels(df)

    # ---------------------------------------------------------
    # Stratified 80/20 split.
    #
    # This ensures LOW, MEDIUM, HIGH, and CRITICAL
    # are represented in both training and test sets.
    # ---------------------------------------------------------

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=risk_labels,
    )

    train_df = train_df.reset_index(drop=True)

    test_df = test_df.reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise ValueError(
            "The stratified split produced an empty "
            "train or test set."
        )

    # ---------------------------------------------------------
    # Calculate the normal baseline using training data only.
    # ---------------------------------------------------------

    normal_training_scenarios = {
        "normal_rest",
        "normal_walking",
        "normal_running",
    }

    normal_train_df = train_df[
        train_df["scenario"].isin(
            normal_training_scenarios
        )
    ].copy()

    if normal_train_df.empty:
        raise ValueError(
            "No normal training rows are available "
            "for the baseline calculation."
        )

    baseline_latitude = float(
        normal_train_df["latitude"].mean()
    )

    baseline_longitude = float(
        normal_train_df["longitude"].mean()
    )

    # ---------------------------------------------------------
    # Create activity encoding.
    # ---------------------------------------------------------

    activity_values = sorted(
        df["activity"]
        .dropna()
        .unique()
        .tolist()
    )

    activity_map = {
        activity: idx
        for idx, activity in enumerate(
            activity_values
        )
    }

    # ---------------------------------------------------------
    # Create model features.
    # ---------------------------------------------------------

    X_train = create_model_features(
        train_df,
        anomaly_model,
        anomaly_scaler,
        activity_map,
        baseline_latitude,
        baseline_longitude,
    )

    X_test = create_model_features(
        test_df,
        anomaly_model,
        anomaly_scaler,
        activity_map,
        baseline_latitude,
        baseline_longitude,
    )

    # ---------------------------------------------------------
    # Create target labels.
    # ---------------------------------------------------------

    y_train = create_risk_labels(train_df)

    y_test = create_risk_labels(test_df)

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training feature rows do not match "
            "the training labels."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test feature rows do not match "
            "the test labels."
        )

    metadata = {
        "risk_class_mapping": {
            key: int(value)
            for key, value in RISK_CLASS_MAPPING.items()
        },
        "feature_names": FEATURE_NAMES,
        "baseline_latitude": baseline_latitude,
        "baseline_longitude": baseline_longitude,
        "split_method": "stratified_train_test_split",
        "test_size": 0.20,
        "random_state": 42,
    }

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        metadata,
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> XGBClassifier:
    """Train an XGBoost classifier with class-balanced sample weights."""

    if X_train.empty:
        raise ValueError(
            "Risk-model training data is empty."
        )

    class_counts = np.bincount(y_train)

    sample_weights = np.ones(
        len(y_train),
        dtype=float,
    )

    num_classes = len(
        RISK_CLASS_MAPPING
    )

    for class_index, count in enumerate(
        class_counts
    ):
        if count > 0:
            sample_weights[
                y_train == class_index
            ] = (
                len(y_train)
                / (num_classes * count)
            )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(
            RISK_CLASS_MAPPING
        ),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=1,
        verbosity=0,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple[
    np.ndarray,
    dict[str, object],
]:
    """Evaluate the risk model using accuracy, precision, recall, F1,
    confusion matrix, and classification report.
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=RISK_CLASS_NAMES,
        zero_division=0,
    )

    per_class_recall = recall_score(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": confusion.tolist(),
        "classification_report": report,
        "per_class_recall": {
            "LOW": float(
                per_class_recall[0]
            ),
            "MEDIUM": float(
                per_class_recall[1]
            ),
            "HIGH": float(
                per_class_recall[2]
            ),
            "CRITICAL": float(
                per_class_recall[3]
            ),
        },
        "predictions": predictions.astype(
            int
        ).tolist(),
    }

    return predictions, metrics


def save_artifacts(
    model: XGBClassifier,
    feature_frame: pd.DataFrame,
    labels: np.ndarray,
    results: dict[str, object],
    metadata: dict[str, object],
) -> None:
    """Persist the trained model, feature store, metadata,
    and evaluation outputs.
    """

    RISK_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RISK_MODEL_PATH.open(
        "wb"
    ) as model_file:
        pickle.dump(
            model,
            model_file,
        )

    RISK_FEATURES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features_payload = {
        "feature_names": FEATURE_NAMES,
        "rows": feature_frame.to_dict(
            orient="records"
        ),
        "labels": labels.astype(
            int
        ).tolist(),
    }

    with RISK_FEATURES_PATH.open(
        "w",
        encoding="utf-8",
    ) as feature_file:
        json.dump(
            features_payload,
            feature_file,
            indent=2,
        )

    RISK_METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RISK_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
        )

    RISK_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RISK_RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as results_file:
        json.dump(
            results,
            results_file,
            indent=2,
        )


def main() -> None:
    """Train and evaluate the XGBoost Silambu risk model
    and save all required artifacts.
    """

    print(
        "Evaluation is based on synthetic data."
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        metadata,
    ) = prepare_data()

    model = train_model(
        X_train,
        y_train,
    )

    predictions, metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # ---------------------------------------------------------
    # Calculate class distributions.
    # ---------------------------------------------------------

    train_distribution = {
        "LOW": int(
            np.sum(y_train == 0)
        ),
        "MEDIUM": int(
            np.sum(y_train == 1)
        ),
        "HIGH": int(
            np.sum(y_train == 2)
        ),
        "CRITICAL": int(
            np.sum(y_train == 3)
        ),
    }

    test_distribution = {
        "LOW": int(
            np.sum(y_test == 0)
        ),
        "MEDIUM": int(
            np.sum(y_test == 1)
        ),
        "HIGH": int(
            np.sum(y_test == 2)
        ),
        "CRITICAL": int(
            np.sum(y_test == 3)
        ),
    }

    # ---------------------------------------------------------
    # Save results.
    # ---------------------------------------------------------

    results = {
        "synthetic_data_note": (
            "Evaluation is based on synthetic data."
        ),
        "training_samples": int(
            len(X_train)
        ),
        "test_samples": int(
            len(X_test)
        ),
        "train_class_distribution": (
            train_distribution
        ),
        "test_class_distribution": (
            test_distribution
        ),
        "accuracy": float(
            metrics["accuracy"]
        ),
        "precision": float(
            metrics["precision"]
        ),
        "recall": float(
            metrics["recall"]
        ),
        "f1_score": float(
            metrics["f1_score"]
        ),
        "per_class_recall": (
            metrics["per_class_recall"]
        ),
        "confusion_matrix": (
            metrics["confusion_matrix"]
        ),
        "classification_report": (
            metrics["classification_report"]
        ),
        "predictions": (
            metrics["predictions"]
        ),
    }

    # ---------------------------------------------------------
    # Console output.
    # ---------------------------------------------------------

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    print(
        "Training class distribution:"
    )

    print(
        train_distribution
    )

    print(
        "Test class distribution:"
    )

    print(
        test_distribution
    )

    print(
        f"Test accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"LOW recall: "
        f"{metrics['per_class_recall']['LOW']:.4f}"
    )

    print(
        f"MEDIUM recall: "
        f"{metrics['per_class_recall']['MEDIUM']:.4f}"
    )

    print(
        f"HIGH recall: "
        f"{metrics['per_class_recall']['HIGH']:.4f}"
    )

    print(
        f"CRITICAL recall: "
        f"{metrics['per_class_recall']['CRITICAL']:.4f}"
    )

    print(
        "Confusion matrix:"
    )

    print(
        np.array(
            metrics["confusion_matrix"]
        )
    )

    print(
        "Classification report:"
    )

    print(
        metrics["classification_report"]
    )

    # ---------------------------------------------------------
    # Save artifacts.
    # ---------------------------------------------------------

    save_artifacts(
        model,
        X_test,
        y_test,
        results,
        metadata,
    )

    print(
        f"Saved risk model to: "
        f"{RISK_MODEL_PATH}"
    )

    print(
        f"Saved risk results to: "
        f"{RISK_RESULTS_PATH}"
    )

    print(
        f"Saved risk features to: "
        f"{RISK_FEATURES_PATH}"
    )

    print(
        f"Saved risk metadata to: "
        f"{RISK_METADATA_PATH}"
    )


if __name__ == "__main__":
    main()