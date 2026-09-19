"""Synthetic sensor data generation for the Silambu child safety wearable prototype.

This module creates a realistic, reproducible dataset that simulates hardware
sensor streams without requiring any physical device.
"""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "silambu_sensor_data.csv"

FIELDNAMES = [
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

SCENARIO_WEIGHTS = {
    "normal_rest": 30,
    "normal_walking": 25,
    "normal_running": 15,
    "sudden_movement": 10,
    "possible_struggle": 8,
    "unusual_location": 6,
    "tampering": 3,
    "emergency_sos": 3,
}

SCENARIO_ORDER = list(SCENARIO_WEIGHTS.keys())


def clamp(value: float, lower: float, upper: float) -> float:
    """Restrict a numeric value to a safe range."""
    return max(lower, min(upper, value))


def scenario_distribution(total_records: int) -> list[str]:
    """Generate contiguous scenario segments that match the intended scenario weights."""
    rng = random.Random(42)

    total_weight = sum(SCENARIO_WEIGHTS.values())
    target_counts = {
        scenario: int(round(total_records * weight / total_weight))
        for scenario, weight in SCENARIO_WEIGHTS.items()
    }
    target_counts["normal_rest"] += total_records - sum(target_counts.values())

    min_segment_lengths = {
        "normal_rest": 20,
        "normal_walking": 20,
        "normal_running": 20,
        "sudden_movement": 8,
        "possible_struggle": 8,
        "unusual_location": 8,
        "tampering": 8,
        "emergency_sos": 8,
    }

    max_segment_lengths = {
        "normal_rest": 180,
        "normal_walking": 180,
        "normal_running": 180,
        "sudden_movement": 60,
        "possible_struggle": 60,
        "unusual_location": 60,
        "tampering": 60,
        "emergency_sos": 60,
    }

    scenario_cycle = [
        "normal_rest",
        "normal_walking",
        "unusual_location",
        "normal_walking",
        "normal_running",
        "sudden_movement",
        "possible_struggle",
        "emergency_sos",
        "normal_rest",
        "normal_walking",
        "tampering",
        "possible_struggle",
        "normal_walking",
    ]

    remaining_counts = target_counts.copy()
    generated_sequence: list[str] = []
    cycle_index = 0

    while sum(remaining_counts.values()) > 0:
        scenario = scenario_cycle[cycle_index % len(scenario_cycle)]
        if remaining_counts[scenario] <= 0:
            cycle_index += 1
            continue

        if remaining_counts[scenario] <= max_segment_lengths[scenario]:
            segment_length = remaining_counts[scenario]
        else:
            segment_length = rng.randint(
                min_segment_lengths[scenario],
                max_segment_lengths[scenario],
            )
            segment_length = min(segment_length, remaining_counts[scenario])

        generated_sequence.extend([scenario] * segment_length)
        remaining_counts[scenario] -= segment_length
        cycle_index += 1

    if len(generated_sequence) != total_records:
        raise ValueError(
            f"Generated scenario sequence length mismatch: expected {total_records}, "
            f"got {len(generated_sequence)}"
        )

    counts = Counter(generated_sequence)
    for scenario in SCENARIO_WEIGHTS:
        if counts.get(scenario, 0) == 0:
            raise ValueError(f"Scenario {scenario} has zero records in the generated sequence.")

    print("Final scenario distribution:")
    for scenario in SCENARIO_ORDER:
        print(f"  {scenario}: {counts.get(scenario, 0)}")

    return generated_sequence


def normal_route_lat_lon(record_index: int, rng: random.Random) -> tuple[float, float]:
    """Create a fictional, safe route with small natural jitter."""
    base_lat = 12.9500 + 0.0035 * math.sin(record_index / 520.0)
    base_lon = 77.5800 + 0.0040 * math.cos(record_index / 680.0)
    lat = base_lat + rng.uniform(-0.0006, 0.0006)
    lon = base_lon + rng.uniform(-0.0008, 0.0008)
    return lat, lon


def unusual_location_lat_lon(record_index: int, rng: random.Random) -> tuple[float, float]:
    """Place the child outside the normal route to simulate an unusual location."""
    base_lat = 12.9500 + 0.0035 * math.sin(record_index / 520.0)
    base_lon = 77.5800 + 0.0040 * math.cos(record_index / 680.0)
    lat = base_lat + 0.085 + rng.uniform(-0.02, 0.02)
    lon = base_lon + 0.120 + rng.uniform(-0.025, 0.025)
    return lat, lon


def generate_sensor_values(scenario: str, record_index: int, rng: random.Random) -> dict[str, float | int]:
    """Generate realistic sensor values that correlate with the selected scenario."""
    wave = math.sin(record_index / 8.0)
    wave2 = math.sin(record_index / 13.0 + 1.5)
    noise = lambda scale: rng.uniform(-scale, scale)

    if scenario == "normal_rest":
        heart_rate = clamp(rng.gauss(68, 4.5), 55, 90)
        spo2 = clamp(rng.gauss(98.5, 0.8), 95, 100)
        accel_x = clamp(0.12 * math.sin(record_index / 17.0) + noise(0.08), -0.8, 0.8)
        accel_y = clamp(0.14 * math.cos(record_index / 19.0) + noise(0.09), -0.9, 0.9)
        accel_z = clamp(0.10 * math.sin(record_index / 21.0) + noise(0.06), -0.7, 0.7)
        gyro_x = clamp(2.0 * math.sin(record_index / 20.0) + noise(1.2), -8, 8)
        gyro_y = clamp(2.8 * math.cos(record_index / 24.0) + noise(1.3), -9, 9)
        gyro_z = clamp(1.8 * math.sin(record_index / 18.0) + noise(1.1), -7, 7)
        speed = clamp(rng.gauss(0.1, 0.2), 0.0, 1.0)

    elif scenario == "normal_walking":
        heart_rate = clamp(rng.gauss(92, 7), 72, 120)
        spo2 = clamp(rng.gauss(98.0, 1.0), 95, 100)
        accel_x = clamp(1.4 * math.sin(record_index / 11.0) + noise(0.3), -2.8, 2.8)
        accel_y = clamp(1.1 * math.cos(record_index / 12.0 + 0.7) + noise(0.25), -2.5, 2.5)
        accel_z = clamp(0.8 * math.sin(record_index / 10.5) + noise(0.2), -1.8, 1.8)
        gyro_x = clamp(18.0 * math.sin(record_index / 10.5) + noise(5.0), -50, 50)
        gyro_y = clamp(14.0 * math.cos(record_index / 11.0) + noise(4.5), -45, 45)
        gyro_z = clamp(9.0 * math.sin(record_index / 9.5) + noise(3.0), -30, 30)
        speed = clamp(rng.gauss(2.9, 0.9), 0.5, 6.0)

    elif scenario == "normal_running":
        heart_rate = clamp(rng.gauss(138, 13), 105, 175)
        spo2 = clamp(rng.gauss(97.5, 1.1), 94, 100)
        accel_x = clamp(2.8 * math.sin(record_index / 6.5) + noise(0.6), -5.0, 5.0)
        accel_y = clamp(2.3 * math.cos(record_index / 7.0 + 0.8) + noise(0.5), -4.6, 4.6)
        accel_z = clamp(1.5 * math.sin(record_index / 6.2 + 0.6) + noise(0.4), -3.5, 3.5)
        gyro_x = clamp(30.0 * math.sin(record_index / 6.2) + noise(8.0), -80, 80)
        gyro_y = clamp(24.0 * math.cos(record_index / 7.1) + noise(7.0), -70, 70)
        gyro_z = clamp(18.0 * math.sin(record_index / 5.8) + noise(6.0), -50, 50)
        speed = clamp(rng.gauss(7.4, 1.8), 3.0, 14.0)

    elif scenario == "sudden_movement":
        heart_rate = clamp(rng.gauss(110, 12), 80, 155)
        spo2 = clamp(rng.gauss(97.0, 1.4), 92, 100)
        spike = 1.0 if (record_index % 21) in {5, 6, 7, 9} else 0.0
        accel_x = clamp(2.5 * math.sin(record_index / 5.0) + spike * 3.0 + noise(0.7), -6.0, 6.0)
        accel_y = clamp(2.0 * math.cos(record_index / 6.2) + spike * 2.8 + noise(0.7), -6.5, 6.5)
        accel_z = clamp(1.4 * math.sin(record_index / 7.1 + 0.6) + spike * 2.2 + noise(0.5), -5.0, 5.0)
        gyro_x = clamp(28.0 * math.sin(record_index / 4.2) + spike * 45.0 + noise(9.0), -100, 100)
        gyro_y = clamp(26.0 * math.cos(record_index / 4.8) + spike * 36.0 + noise(8.0), -100, 100)
        gyro_z = clamp(20.0 * math.sin(record_index / 5.5) + spike * 30.0 + noise(7.0), -90, 90)
        speed = clamp(rng.gauss(4.5, 2.1), 0.7, 12.0)

    elif scenario == "possible_struggle":
        heart_rate = clamp(rng.gauss(128, 17), 96, 180)
        spo2 = clamp(rng.gauss(94.5, 2.1), 88, 99)
        accel_x = clamp(2.8 * math.sin(record_index / 4.2 + 0.8) + rng.uniform(-2.0, 2.2), -6.2, 6.2)
        accel_y = clamp(2.4 * math.cos(record_index / 5.1 + 1.1) + rng.uniform(-1.8, 1.9), -6.0, 6.0)
        accel_z = clamp(1.8 * math.sin(record_index / 3.8) + rng.uniform(-1.5, 1.3), -5.0, 5.0)
        gyro_x = clamp(45.0 * math.sin(record_index / 3.5) + rng.uniform(-25, 25), -110, 110)
        gyro_y = clamp(38.0 * math.cos(record_index / 4.0 + 0.7) + rng.uniform(-22, 22), -105, 105)
        gyro_z = clamp(32.0 * math.sin(record_index / 4.5) + rng.uniform(-20, 20), -95, 95)
        speed = clamp(rng.gauss(5.6, 2.6), 1.0, 13.0)

    elif scenario == "unusual_location":
        heart_rate = clamp(rng.gauss(86, 7), 65, 120)
        spo2 = clamp(rng.gauss(97.8, 1.2), 94, 100)
        accel_x = clamp(0.9 * math.sin(record_index / 12.0) + noise(0.22), -2.2, 2.2)
        accel_y = clamp(0.8 * math.cos(record_index / 13.5 + 1.0) + noise(0.2), -2.0, 2.0)
        accel_z = clamp(0.7 * math.sin(record_index / 9.0) + noise(0.18), -1.8, 1.8)
        gyro_x = clamp(7.0 * math.sin(record_index / 11.0) + noise(2.6), -20, 20)
        gyro_y = clamp(8.5 * math.cos(record_index / 12.5) + noise(2.8), -22, 22)
        gyro_z = clamp(5.0 * math.sin(record_index / 10.5 + 0.5) + noise(2.3), -18, 18)
        speed = clamp(rng.gauss(3.3, 1.4), 0.5, 9.0)

    elif scenario == "tampering":
        heart_rate = clamp(rng.gauss(118, 15), 90, 170)
        spo2 = clamp(rng.gauss(95.5, 2.0), 88, 99)
        accel_x = clamp(3.5 * math.sin(record_index / 5.0) + 4.0 * math.sin(record_index / 2.5) + noise(1.0), -8.0, 8.0)
        accel_y = clamp(3.0 * math.cos(record_index / 4.1) + 3.2 * math.sin(record_index / 2.9) + noise(0.9), -8.0, 8.0)
        accel_z = clamp(2.4 * math.sin(record_index / 4.8 + 1.3) + 2.8 * math.cos(record_index / 2.4) + noise(0.7), -7.0, 7.0)
        gyro_x = clamp(50.0 * math.sin(record_index / 3.8) + rng.uniform(-25, 25), -120, 120)
        gyro_y = clamp(44.0 * math.cos(record_index / 4.2 + 0.2) + rng.uniform(-24, 24), -115, 115)
        gyro_z = clamp(38.0 * math.sin(record_index / 4.7) + rng.uniform(-22, 22), -100, 100)
        speed = clamp(rng.gauss(6.2, 2.2), 1.5, 15.0)

    elif scenario == "emergency_sos":
        heart_rate = clamp(rng.gauss(165, 18), 125, 210)
        spo2 = clamp(rng.gauss(90.0, 3.0), 80, 97)
        accel_x = clamp(2.2 * math.sin(record_index / 6.0) + 2.0 * math.sin(record_index / 2.2) + noise(0.8), -5.5, 5.5)
        accel_y = clamp(2.0 * math.cos(record_index / 5.6) + 1.8 * math.sin(record_index / 2.4) + noise(0.8), -5.2, 5.2)
        accel_z = clamp(1.7 * math.sin(record_index / 5.9 + 0.9) + noise(0.6), -4.0, 4.0)
        gyro_x = clamp(38.0 * math.sin(record_index / 3.9) + 18.0 * math.sin(record_index / 1.7) + noise(8.0), -100, 100)
        gyro_y = clamp(32.0 * math.cos(record_index / 4.2) + 16.0 * math.sin(record_index / 1.9) + noise(8.0), -100, 100)
        gyro_z = clamp(26.0 * math.sin(record_index / 4.5 + 0.7) + noise(7.0), -80, 80)
        speed = clamp(rng.gauss(8.8, 3.0), 2.0, 20.0)

    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    return {
        "heart_rate": round(clamp(float(heart_rate), 40, 220), 2),
        "spo2": round(clamp(float(spo2), 70, 100), 2),
        "accel_x": round(clamp(float(accel_x), -12.0, 12.0), 3),
        "accel_y": round(clamp(float(accel_y), -12.0, 12.0), 3),
        "accel_z": round(clamp(float(accel_z), -12.0, 12.0), 3),
        "gyro_x": round(clamp(float(gyro_x), -180.0, 180.0), 3),
        "gyro_y": round(clamp(float(gyro_y), -180.0, 180.0), 3),
        "gyro_z": round(clamp(float(gyro_z), -180.0, 180.0), 3),
        "speed": round(clamp(float(speed), 0.0, 25.0), 3),
    }


def build_record(record_index: int, scenario: str, rng: random.Random, start_time: datetime) -> dict[str, object]:
    """Build a single sensor record with correlated values and metadata."""
    timestamp = (start_time + timedelta(seconds=12 * record_index)).strftime("%Y-%m-%d %H:%M:%S")

    if scenario == "unusual_location":
        latitude, longitude = unusual_location_lat_lon(record_index, rng)
    else:
        latitude, longitude = normal_route_lat_lon(record_index, rng)

    sensor_values = generate_sensor_values(scenario, record_index, rng)

    activity_map = {
        "normal_rest": "resting",
        "normal_walking": "walking",
        "normal_running": "running",
        "sudden_movement": "sudden_movement",
        "possible_struggle": "struggle",
        "unusual_location": "travel",
        "tampering": "tampering",
        "emergency_sos": "emergency",
    }

    tamper_value = 1 if scenario == "tampering" else 0
    sos_value = 1 if scenario == "emergency_sos" else 0

    return {
        "timestamp": timestamp,
        "heart_rate": sensor_values["heart_rate"],
        "spo2": sensor_values["spo2"],
        "accel_x": sensor_values["accel_x"],
        "accel_y": sensor_values["accel_y"],
        "accel_z": sensor_values["accel_z"],
        "gyro_x": sensor_values["gyro_x"],
        "gyro_y": sensor_values["gyro_y"],
        "gyro_z": sensor_values["gyro_z"],
        "latitude": round(latitude, 6),
        "longitude": round(longitude, 6),
        "speed": sensor_values["speed"],
        "activity": activity_map[scenario],
        "tamper": tamper_value,
        "sos": sos_value,
        "scenario": scenario,
    }


def generate_dataset(num_records: int = 50_000, seed: int = 42) -> list[dict[str, object]]:
    """Generate a realistic synthetic dataset for the wearable prototype."""
    rng = random.Random(seed)
    start_time = datetime(2026, 1, 1, 7, 0, 0)
    scenario_sequence = scenario_distribution(num_records)

    return [
        build_record(index, scenario_sequence[index], rng, start_time)
        for index in range(num_records)
    ]


def write_dataset(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write the generated dataset to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_distribution(rows: list[dict[str, object]]) -> None:
    """Print the scenario distribution for the generated data."""
    counts = Counter(row["scenario"] for row in rows)
    print("Scenario distribution:")
    for scenario in SCENARIO_ORDER:
        print(f"  {scenario}: {counts.get(scenario, 0)}")


def print_basic_statistics(rows: list[dict[str, object]]) -> None:
    """Print a concise statistical summary for key numeric sensor fields."""
    print("\nBasic statistics:")
    numeric_fields = [
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
    ]

    for field in numeric_fields:
        values = [float(row[field]) for row in rows]
        print(
            f"  {field}: min={min(values):.3f}, mean={mean(values):.3f}, "
            f"median={median(values):.3f}, max={max(values):.3f}"
        )


def main() -> None:
    """Generate and save the dataset, then print a short summary."""
    rows = generate_dataset(num_records=50_000, seed=42)
    write_dataset(rows, OUTPUT_PATH)

    print(f"Number of records: {len(rows)}")
    print_distribution(rows)
    print("\nFirst 5 rows:")
    for row in rows[:5]:
        print(row)
    print_basic_statistics(rows)
    print(f"\nDataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
