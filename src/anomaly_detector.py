"""Isolation Forest anomaly detector for Silambu normal-behavior learning.

This prototype keeps the detector unsupervised and learns temporal sensor behavior
from normal windows only, while evaluating held-out normal and safety-event windows.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "silambu_sensor_data.csv"
ANOMALY_SCALER_PATH = PROJECT_ROOT / "models" / "anomaly_scaler.pkl"
MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_detector.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "anomaly_metadata.json"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "anomaly_detection_results.json"

NORMAL_SCENARIOS = {
    "normal_rest",
    "normal_walking",
    "normal_running",
}
ANOMALY_SCENARIOS = {
    "sudden_movement",
    "possible_struggle",
    "unusual_location",
    "tampering",
    "emergency_sos",
}
WINDOW_LENGTH = 30
CONTAMINATION_VALUES = [0.03, 0.05, 0.08, 0.10]

WINDOW_FEATURE_COLUMNS = [
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


def load_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Silambu sensor dataset."""
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("The raw dataset is empty.")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the derived sensor features used in temporal window statistics."""
    df = df.copy()
    df["acceleration_magnitude"] = np.sqrt(
        df["accel_x"] ** 2 + df["accel_y"] ** 2 + df["accel_z"] ** 2
    )
    df["gyroscope_magnitude"] = np.sqrt(
        df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2
    )
    df["heart_rate_change"] = df["heart_rate"].diff().fillna(0.0).abs()
    df["speed_change"] = df["speed"].diff().fillna(0.0).abs()
    return df


def calculate_location_features(
    df: pd.DataFrame,
    baseline_latitude: float,
    baseline_longitude: float,
) -> pd.DataFrame:
    """Add baseline-relative location metrics without using raw coordinates in IF inputs."""
    df = df.copy()
    df["latitude_delta"] = (df["latitude"] - baseline_latitude).abs()
    df["longitude_delta"] = (df["longitude"] - baseline_longitude).abs()
    df["distance_from_baseline_location"] = np.hypot(
        df["latitude"] - baseline_latitude,
        df["longitude"] - baseline_longitude,
    )
    return df


def create_windows(df: pd.DataFrame, window_length: int = WINDOW_LENGTH) -> list[dict[str, object]]:
    """Create chronological windows of consecutive sensor readings."""
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if len(df) < window_length:
        raise ValueError(f"Dataset is shorter than the required window length of {window_length} readings.")

    windows: list[dict[str, object]] = []
    for start in range(0, len(df) - window_length + 1):
        window_df = df.iloc[start : start + window_length].copy()
        safety_count = int(window_df["scenario"].isin(ANOMALY_SCENARIOS).sum())
        window_label = 1 if safety_count > 0 else 0
        windows.append(
            {
                "start_index": start,
                "end_index": start + window_length - 1,
                "label": window_label,
                "scenarios": window_df["scenario"].dropna().unique().tolist(),
                "data": window_df,
            }
        )
    return windows


def build_window_features(
    window_df: pd.DataFrame,
    baseline_latitude: float,
    baseline_longitude: float,
) -> dict[str, float | int]:
    """Aggregate a 30-reading window into behavioral and location statistics."""
    acceleration_magnitude = np.sqrt(
        window_df["accel_x"] ** 2 + window_df["accel_y"] ** 2 + window_df["accel_z"] ** 2
    )
    gyroscope_magnitude = np.sqrt(
        window_df["gyro_x"] ** 2 + window_df["gyro_y"] ** 2 + window_df["gyro_z"] ** 2
    )
    heart_rate_change = window_df["heart_rate"].diff().fillna(0.0).abs()
    speed_change = window_df["speed"].diff().fillna(0.0).abs()

    latitude = window_df["latitude"].astype(float)
    longitude = window_df["longitude"].astype(float)
    lat_disp = abs(latitude.iloc[-1] - latitude.iloc[0])
    lon_disp = abs(longitude.iloc[-1] - longitude.iloc[0])
    total_approx_displacement = float(np.hypot(lat_disp, lon_disp))
    dist_from_baseline = np.hypot(
        latitude.mean() - baseline_latitude,
        longitude.mean() - baseline_longitude,
    )
    max_dist_from_baseline = float(
        np.max(np.hypot(latitude - baseline_latitude, longitude - baseline_longitude))
    )

    feature_row = {
        "mean_acceleration_magnitude": float(acceleration_magnitude.mean()),
        "std_acceleration_magnitude": float(acceleration_magnitude.std(ddof=0)),
        "max_acceleration_magnitude": float(acceleration_magnitude.max()),
        "mean_gyroscope_magnitude": float(gyroscope_magnitude.mean()),
        "std_gyroscope_magnitude": float(gyroscope_magnitude.std(ddof=0)),
        "max_gyroscope_magnitude": float(gyroscope_magnitude.max()),
        "mean_heart_rate": float(window_df["heart_rate"].mean()),
        "std_heart_rate": float(window_df["heart_rate"].std(ddof=0)),
        "min_heart_rate": float(window_df["heart_rate"].min()),
        "max_heart_rate": float(window_df["heart_rate"].max()),
        "heart_rate_range": float(window_df["heart_rate"].max() - window_df["heart_rate"].min()),
        "mean_absolute_heart_rate_change": float(heart_rate_change.mean()),
        "mean_speed": float(window_df["speed"].mean()),
        "std_speed": float(window_df["speed"].std(ddof=0)),
        "max_speed": float(window_df["speed"].max()),
        "speed_range": float(window_df["speed"].max() - window_df["speed"].min()),
        "mean_absolute_speed_change": float(speed_change.mean()),
        "latitude_displacement": float(lat_disp),
        "longitude_displacement": float(lon_disp),
        "total_approximate_displacement": total_approx_displacement,
        "distance_from_normal_baseline": float(dist_from_baseline),
        "maximum_distance_from_normal_baseline": max_dist_from_baseline,
        "mean_spo2": float(window_df["spo2"].mean()),
        "min_spo2": float(window_df["spo2"].min()),
        "tamper_count": int(window_df["tamper"].sum()),
        "sos_count": int(window_df["sos"].sum()),
    }
    return feature_row


def prepare_train_test(df: pd.DataFrame) -> dict[str, object]:
    """Create train/test windows using only normal windows for training and both normal/safety windows for testing."""
    df = create_features(df).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    normal_df = df[df["scenario"].isin(NORMAL_SCENARIOS)].copy()
    safety_df = df[df["scenario"].isin(ANOMALY_SCENARIOS)].copy()

    if normal_df.empty:
        raise ValueError("No normal training samples were found in the dataset.")
    if safety_df.empty:
        raise ValueError("No safety-event samples were found for evaluation.")

    normal_train_cut = int(len(normal_df) * 0.8)
    normal_train_df = normal_df.iloc[:normal_train_cut].copy()
    normal_test_df = normal_df.iloc[normal_train_cut:].copy()

    if len(normal_train_df) < WINDOW_LENGTH:
        raise ValueError("Not enough normal training data to build a 30-reading training window.")
    if len(normal_test_df) < WINDOW_LENGTH:
        raise ValueError("Not enough normal test data to build a 30-reading test window.")

    baseline_latitude = float(normal_train_df["latitude"].mean())
    baseline_longitude = float(normal_train_df["longitude"].mean())

    normal_train_df = calculate_location_features(normal_train_df, baseline_latitude, baseline_longitude)
    normal_test_df = calculate_location_features(normal_test_df, baseline_latitude, baseline_longitude)
    safety_df = calculate_location_features(safety_df, baseline_latitude, baseline_longitude)

    train_windows = create_windows(normal_train_df, WINDOW_LENGTH)
    normal_test_windows = create_windows(normal_test_df, WINDOW_LENGTH)
    safety_windows = create_windows(safety_df, WINDOW_LENGTH)

    train_rows = []
    train_labels = []
    for window in train_windows:
        train_rows.append(build_window_features(window["data"], baseline_latitude, baseline_longitude))
        train_labels.append(int(window["label"]))

    normal_test_rows = []
    normal_test_labels = []
    for window in normal_test_windows:
        normal_test_rows.append(build_window_features(window["data"], baseline_latitude, baseline_longitude))
        normal_test_labels.append(int(window["label"]))

    anomaly_test_rows = []
    anomaly_test_labels = []
    for window in safety_windows:
        anomaly_test_rows.append(build_window_features(window["data"], baseline_latitude, baseline_longitude))
        anomaly_test_labels.append(int(window["label"]))

    X_train = pd.DataFrame(train_rows, columns=WINDOW_FEATURE_COLUMNS)
    y_train = np.asarray(train_labels, dtype=int)
    X_normal_test = pd.DataFrame(normal_test_rows, columns=WINDOW_FEATURE_COLUMNS)
    y_normal_test = np.asarray(normal_test_labels, dtype=int)
    X_anomaly_test = pd.DataFrame(anomaly_test_rows, columns=WINDOW_FEATURE_COLUMNS)
    y_anomaly_test = np.asarray(anomaly_test_labels, dtype=int)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=WINDOW_FEATURE_COLUMNS,
        index=X_train.index,
    )
    X_normal_test_scaled = pd.DataFrame(
        scaler.transform(X_normal_test),
        columns=WINDOW_FEATURE_COLUMNS,
        index=X_normal_test.index,
    )
    X_anomaly_test_scaled = pd.DataFrame(
        scaler.transform(X_anomaly_test),
        columns=WINDOW_FEATURE_COLUMNS,
        index=X_anomaly_test.index,
    )

    with ANOMALY_SCALER_PATH.open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    scenario_window_map: dict[str, list[dict[str, object]]] = {scenario: [] for scenario in sorted(ANOMALY_SCENARIOS)}
    for window in safety_windows:
        window_scenarios = set(window["scenarios"])
        for scenario in window_scenarios:
            if scenario in scenario_window_map:
                scenario_window_map[scenario].append(window)

    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_normal_test": X_normal_test_scaled,
        "y_normal_test": y_normal_test,
        "X_anomaly_test": X_anomaly_test_scaled,
        "y_anomaly_test": y_anomaly_test,
        "scaler": scaler,
        "baseline_latitude": baseline_latitude,
        "baseline_longitude": baseline_longitude,
        "normal_train_windows": len(train_windows),
        "normal_test_windows": len(normal_test_windows),
        "anomaly_test_windows": len(safety_windows),
        "feature_names": WINDOW_FEATURE_COLUMNS,
        "scenario_window_map": scenario_window_map,
        "safety_windows": safety_windows,
    }


def train_detector(X_train: pd.DataFrame, contamination: float) -> IsolationForest:
    """Train an Isolation Forest on scaled normal windows only."""
    if X_train.empty:
        raise ValueError("Training data for Isolation Forest is empty.")
    model = IsolationForest(contamination=float(contamination), random_state=42)
    model.fit(X_train)
    return model


def evaluate_detector(
    model: IsolationForest,
    X_test: pd.DataFrame,
    true_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | str | list[int] | list[float]]]:
    """Evaluate the detector given a test set and ground-truth labels."""
    decision_scores = model.decision_function(X_test)
    predictions = model.predict(X_test)
    binary_predictions = np.where(predictions == -1, 1, 0).astype(int)

    normal_mask = true_labels == 0
    anomaly_mask = true_labels == 1
    normal_detection_rate = (
        float(np.mean(binary_predictions[normal_mask] == 0)) if np.any(normal_mask) else 0.0
    )
    anomaly_detection_rate = (
        float(np.mean(binary_predictions[anomaly_mask] == 1)) if np.any(anomaly_mask) else 0.0
    )
    balanced_detection_score = float(np.mean([normal_detection_rate, anomaly_detection_rate]))
    accuracy = float(accuracy_score(true_labels, binary_predictions))
    cm = confusion_matrix(true_labels, binary_predictions, labels=[0, 1])
    report = classification_report(
        true_labels,
        binary_predictions,
        target_names=["normal", "anomaly"],
        zero_division=0,
    )

    metrics = {
        "normal_detection_rate": normal_detection_rate,
        "anomaly_detection_rate": anomaly_detection_rate,
        "balanced_detection_score": balanced_detection_score,
        "overall_accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "decision_scores": decision_scores.tolist(),
        "predictions": binary_predictions.tolist(),
    }
    return binary_predictions, metrics


def tune_contamination(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple[IsolationForest, float, list[dict[str, object]]]:
    """Tune contamination via the mean of the normal and anomaly detection rates."""
    candidate_results: list[dict[str, object]] = []
    best_result: dict[str, object] | None = None

    for contamination in CONTAMINATION_VALUES:
        model = train_detector(X_train, contamination)
        predictions, metrics = evaluate_detector(model, X_test, y_test)
        result = {
            "contamination": contamination,
            "model": model,
            "predictions": predictions,
            "metrics": metrics,
        }
        candidate_results.append(result)
        candidate_score = float(metrics["balanced_detection_score"])
        if best_result is None or candidate_score > float(best_result["metrics"]["balanced_detection_score"]):
            best_result = result

        print(
            f"contamination={contamination:.2f} | "
            f"normal_detection_rate={metrics['normal_detection_rate']:.4f} | "
            f"anomaly_detection_rate={metrics['anomaly_detection_rate']:.4f} | "
            f"balanced_detection_score={metrics['balanced_detection_score']:.4f} | "
            f"overall_accuracy={metrics['overall_accuracy']:.4f}"
        )

    if best_result is None:
        raise ValueError("No contamination tuning result was produced.")

    return best_result["model"], float(best_result["contamination"]), candidate_results


def save_artifacts(
    model: IsolationForest,
    scaler: StandardScaler,
    metadata: dict[str, object],
    results: dict[str, object],
    model_path: Path = MODEL_PATH,
    scaler_path: Path = ANOMALY_SCALER_PATH,
    metadata_path: Path = METADATA_PATH,
    results_path: Path = RESULTS_PATH,
) -> None:
    """Persist the selected model, scaler, metadata, and evaluation results."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with scaler_path.open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)


def main() -> None:
    """Train a normal-only Isolation Forest and evaluate chronology-aware temporal windows."""
    df = load_data()
    prepared = prepare_train_test(df)

    X_train = prepared["X_train"]
    y_train = prepared["y_train"]
    X_normal_test = prepared["X_normal_test"]
    y_normal_test = prepared["y_normal_test"]
    X_anomaly_test = prepared["X_anomaly_test"]
    y_anomaly_test = prepared["y_anomaly_test"]
    baseline_latitude = float(prepared["baseline_latitude"])
    baseline_longitude = float(prepared["baseline_longitude"])
    normal_train_windows = int(prepared["normal_train_windows"])
    normal_test_windows = int(prepared["normal_test_windows"])
    anomaly_test_windows = int(prepared["anomaly_test_windows"])
    feature_names = list(prepared["feature_names"])
    scenario_window_map = prepared["scenario_window_map"]
    safety_windows = prepared["safety_windows"]
    scaler = prepared["scaler"]

    X_combined_test = pd.concat([X_normal_test, X_anomaly_test], ignore_index=True)
    y_combined_test = np.concatenate([y_normal_test, y_anomaly_test])

    best_model, selected_contamination, candidate_results = tune_contamination(
        X_train,
        y_train,
        X_combined_test,
        y_combined_test,
    )

    final_predictions, final_metrics = evaluate_detector(best_model, X_combined_test, y_combined_test)
    final_metrics["normal_test_windows"] = normal_test_windows
    final_metrics["anomaly_test_windows"] = anomaly_test_windows

    scenario_rates: dict[str, dict[str, float]] = {}
    for scenario in sorted(ANOMALY_SCENARIOS):
        scenario_windows = scenario_window_map.get(scenario, [])
        if not scenario_windows:
            scenario_rates[scenario] = {
                "behavioral_anomaly_detection_rate": 0.0,
                "direct_tamper_alert_detection_rate": 0.0,
                "direct_sos_alert_detection_rate": 0.0,
            }
            continue

        scenario_rows = []
        for window in scenario_windows:
            scenario_rows.append(
                build_window_features(window["data"], baseline_latitude, baseline_longitude)
            )
        scenario_df = pd.DataFrame(scenario_rows, columns=WINDOW_FEATURE_COLUMNS)
        scenario_features = pd.DataFrame(
            scaler.transform(scenario_df),
            columns=WINDOW_FEATURE_COLUMNS,
            index=scenario_df.index,
        )
        scenario_predictions = np.where(best_model.predict(scenario_features) == -1, 1, 0).astype(int)
        scenario_anomaly_rate = float(np.mean(scenario_predictions == 1))
        direct_tamper_rate = float(
            np.mean([int(window["data"]["tamper"].sum() > 0) for window in scenario_windows])
        )
        direct_sos_rate = float(
            np.mean([int(window["data"]["sos"].sum() > 0) for window in scenario_windows])
        )
        scenario_rates[scenario] = {
            "behavioral_anomaly_detection_rate": scenario_anomaly_rate,
            "direct_tamper_alert_detection_rate": direct_tamper_rate if scenario == "tampering" else 0.0,
            "direct_sos_alert_detection_rate": direct_sos_rate if scenario == "emergency_sos" else 0.0,
        }

    direct_tamper_alert_rate = float(
        np.mean([int(window["data"]["tamper"].sum() > 0) for window in safety_windows])
    )
    direct_sos_alert_rate = float(
        np.mean([int(window["data"]["sos"].sum() > 0) for window in safety_windows])
    )

    results = {
        "synthetic_data_note": "Evaluation is based on synthetic data.",
        "window_length": WINDOW_LENGTH,
        "training_windows": normal_train_windows,
        "normal_test_windows": normal_test_windows,
        "anomaly_test_windows": anomaly_test_windows,
        "selected_contamination": selected_contamination,
        "normal_detection_rate": float(final_metrics["normal_detection_rate"]),
        "anomaly_detection_rate": float(final_metrics["anomaly_detection_rate"]),
        "balanced_detection_score": float(final_metrics["balanced_detection_score"]),
        "overall_accuracy": float(final_metrics["overall_accuracy"]),
        "confusion_matrix": final_metrics["confusion_matrix"],
        "classification_report": final_metrics["classification_report"],
        "direct_tamper_alert_rate": direct_tamper_alert_rate,
        "direct_sos_alert_rate": direct_sos_alert_rate,
        "per_scenario_detection_rates": scenario_rates,
        "candidate_contamination_results": [
            {
                "contamination": candidate["contamination"],
                "normal_detection_rate": candidate["metrics"]["normal_detection_rate"],
                "anomaly_detection_rate": candidate["metrics"]["anomaly_detection_rate"],
                "balanced_detection_score": candidate["metrics"]["balanced_detection_score"],
                "overall_accuracy": candidate["metrics"]["overall_accuracy"],
            }
            for candidate in candidate_results
        ],
    }

    metadata = {
        "window_length": WINDOW_LENGTH,
        "selected_contamination": selected_contamination,
        "feature_names": feature_names,
        "normal_baseline_latitude": baseline_latitude,
        "normal_baseline_longitude": baseline_longitude,
    }

    print("Evaluation is based on synthetic data.")
    print(f"Number of training windows: {normal_train_windows}")
    print(f"Number of normal test windows: {normal_test_windows}")
    print(f"Number of anomaly test windows: {anomaly_test_windows}")
    print(f"Selected contamination: {selected_contamination:.2f}")
    print(f"Normal detection rate: {results['normal_detection_rate']:.4f}")
    print(f"Anomaly detection rate: {results['anomaly_detection_rate']:.4f}")
    print(f"Balanced detection score: {results['balanced_detection_score']:.4f}")
    print(f"Overall accuracy: {results['overall_accuracy']:.4f}")
    print("Confusion matrix:")
    print(np.array(results["confusion_matrix"]))
    print("Classification report:")
    print(results["classification_report"])
    print("Per-scenario detection rates:")
    for scenario, metrics in scenario_rates.items():
        print(f"  {scenario}: {metrics}")

    save_artifacts(best_model, scaler, metadata, results)
    print(f"Saved anomaly detector to: {MODEL_PATH}")
    print(f"Saved anomaly scaler to: {ANOMALY_SCALER_PATH}")
    print(f"Saved anomaly metadata to: {METADATA_PATH}")
    print(f"Saved evaluation results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
