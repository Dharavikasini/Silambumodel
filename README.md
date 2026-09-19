# Silambu AI Model

AI/ML module for the **Silambu Child Safety System**.

Silambu is a child-safety solution designed to analyze wearable sensor data and identify changes in activity, behavior, anomalies, and overall risk.

This repository contains the complete ML pipeline required for inference and is intended to be integrated with a **Python backend**, **PostgreSQL database**, and **React Native mobile application**.

---

# Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. System Architecture](#2-system-architecture)
- [3. AI Pipeline](#3-ai-pipeline)
- [4. Models](#4-models)
- [5. Input Data](#5-input-data)
- [6. Sequence Processing](#6-sequence-processing)
- [7. Feature Engineering](#7-feature-engineering)
- [8. Model Outputs](#8-model-outputs)
- [9. Project Structure](#9-project-structure)
- [10. Requirements](#10-requirements)
- [11. Installation](#11-installation)
- [12. Running the Existing Pipeline](#12-running-the-existing-pipeline)
- [13. Docker](#13-docker)
- [14. Backend Integration](#14-backend-integration)
- [15. API Contract](#15-api-contract)
- [16. FastAPI Integration Example](#16-fastapi-integration-example)
- [17. PostgreSQL Integration](#17-postgresql-integration)
- [18. React Native Integration](#18-react-native-integration)
- [19. Prediction Flow](#19-prediction-flow)
- [20. Model Files](#20-model-files)
- [21. Important Integration Rules](#21-important-integration-rules)
- [22. Testing](#22-testing)
- [23. Model Performance](#23-model-performance)
- [24. Current Status](#24-current-status)
- [25. Future Improvements](#25-future-improvements)
- [26. Developer Handoff](#26-developer-handoff)
- [27. Disclaimer](#27-disclaimer)

---

# 1. Project Overview

The Silambu AI system processes wearable sensor data and produces four levels of intelligence:

1. **Activity Recognition**
2. **Behavior Analysis**
3. **Anomaly Detection**
4. **Risk Prediction**

The AI pipeline works on a temporal sequence of sensor readings rather than a single reading.

The current pipeline uses a **30-reading window** for inference.

```text
Wearable Sensor Data
        |
        v
Data Validation
        |
        v
Feature Engineering
        |
        v
30-Reading Temporal Window
        |
        +----------------------+
        |                      |
        v                      v
 Activity Model         Behavior Model
 DeepConvLSTM                 LSTM
        |                      |
        +----------+-----------+
                   |
                   v
            Anomaly Detection
            Isolation Forest
                   |
                   v
             Risk Prediction
                 XGBoost
                   |
                   v
            Final AI Result
```

The AI model is designed to run as part of the **Python backend**.

The React Native application should communicate with the backend through an API instead of directly loading the ML models.

---

# 2. System Architecture

The overall application is expected to follow this architecture:

```text
                    SILAMBU SYSTEM
                         |
          +--------------+--------------+
          |                             |
          v                             v
   React Native                    Python Backend
   Mobile App                           |
          |                             |
          | REST API                    |
          +---------------------------->|
                                        |
                                        v
                                Silambu AI Module
                                        |
                   +--------------------+--------------------+
                   |                    |                    |
                   v                    v                    v
             Activity             Behavior             Anomaly
             DeepConvLSTM             LSTM            Isolation Forest
                   |                    |                    |
                   +--------------------+--------------------+
                                        |
                                        v
                                  Risk Prediction
                                     XGBoost
                                        |
                                        v
                                  AI Prediction
                                        |
                                        v
                                   PostgreSQL
```

## Component Responsibilities

### React Native

The React Native application is responsible for:

- Mobile UI
- Child monitoring dashboard
- Activity display
- Behavior display
- Risk display
- Location display
- Alert display
- Communication with the Python backend

### Python Backend

The Python backend is responsible for:

- API endpoints
- Sensor data ingestion
- Input validation
- Maintaining the 30-reading window
- Calling the AI models
- PostgreSQL communication
- Storing sensor readings
- Storing AI predictions
- Alert and notification logic
- Communication with React Native

### Silambu AI

The AI module is responsible for:

- Feature engineering
- Activity recognition
- Behavior analysis
- Anomaly detection
- Risk prediction

### PostgreSQL

PostgreSQL is responsible for persistent storage of:

- Sensor readings
- AI predictions
- Child/device information
- Historical data

---

# 3. AI Pipeline

The Silambu AI pipeline consists of four major components.

```text
                    SENSOR DATA
                         |
                         v
                 INPUT VALIDATION
                         |
                         v
                FEATURE ENGINEERING
                         |
                         v
                30-READING WINDOW
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
      Activity        Behavior       Anomaly
      Recognition     Analysis       Detection
          |              |              |
          +--------------+--------------+
                         |
                         v
                  Risk Prediction
                         |
                         v
                  Final Prediction
```

---

# 4. Models

## 4.1 Activity Recognition

### Model

**DeepConvLSTM**

### Purpose

Identifies the physical activity represented by the current sensor sequence.

### Supported activities

```text
resting
walking
running
sudden_movement
struggle
travel
tampering
emergency
```

### Input

The activity model uses a 30-reading temporal window with 11 features:

```text
heart_rate
spo2
accel_x
accel_y
accel_z
gyro_x
gyro_y
gyro_z
speed
acceleration_magnitude
gyroscope_magnitude
```

### Output

Example:

```text
Activity: walking
Confidence: 0.98
```

---

## 4.2 Behavior Analysis

### Model

**LSTM**

### Purpose

Analyzes temporal sensor patterns to identify the current behavioral state.

### Supported behavior classes

```text
resting
walking
running
sudden_movement
struggle
travel
tampering
emergency
```

### Input Features

The behavior model uses:

```text
heart_rate
spo2
accel_x
accel_y
accel_z
gyro_x
gyro_y
gyro_z
speed
acceleration_magnitude
gyroscope_magnitude
heart_rate_change
speed_change
```

### Output

Example:

```text
Behavior: struggle
Confidence: 0.94
```

---

## 4.3 Anomaly Detection

### Model

**Isolation Forest**

### Purpose

Detects sensor patterns that differ from normal behavioral patterns.

The anomaly detector works with statistical features calculated from a 30-reading window.

### Feature groups

The anomaly detector uses features related to:

- Acceleration statistics
- Gyroscope statistics
- Heart-rate statistics
- Speed statistics
- GPS displacement
- Distance from normal baseline
- SpO₂
- Tamper events
- SOS events

### Output

```text
Anomaly Score
Anomaly Detected: True / False
```

Example:

```json
{
  "detected": true,
  "score": 0.126
}
```

The anomaly score is a model score and should not automatically be interpreted as a probability.

---

## 4.4 Risk Prediction

### Model

**XGBoost**

### Purpose

Determines the overall risk category using sensor-derived features and outputs from the AI pipeline.

### Risk Levels

```text
LOW
MEDIUM
HIGH
CRITICAL
```

### Current Risk Mapping

| Scenario | Risk Level |
|---|---|
| normal_rest | LOW |
| normal_walking | LOW |
| normal_running | LOW |
| unusual_location | MEDIUM |
| sudden_movement | HIGH |
| tampering | HIGH |
| possible_struggle | CRITICAL |
| emergency_sos | CRITICAL |

---

# 5. Input Data

The AI pipeline expects sensor readings containing the following fields:

```text
timestamp

heart_rate
spo2

accel_x
accel_y
accel_z

gyro_x
gyro_y
gyro_z

latitude
longitude
speed

tamper
sos
```

## Sensor Field Description

| Field | Description |
|---|---|
| timestamp | Time at which the reading was recorded |
| heart_rate | Heart-rate reading |
| spo2 | Blood oxygen saturation |
| accel_x | Accelerometer X-axis |
| accel_y | Accelerometer Y-axis |
| accel_z | Accelerometer Z-axis |
| gyro_x | Gyroscope X-axis |
| gyro_y | Gyroscope Y-axis |
| gyro_z | Gyroscope Z-axis |
| latitude | GPS latitude |
| longitude | GPS longitude |
| speed | Current movement speed |
| tamper | Tamper sensor status |
| sos | SOS button status |

---

## Example Sensor Reading

```json
{
  "timestamp": "2026-09-19T12:30:00",
  "heart_rate": 112,
  "spo2": 97,
  "accel_x": 2.31,
  "accel_y": 1.12,
  "accel_z": 9.81,
  "gyro_x": 0.41,
  "gyro_y": 1.22,
  "gyro_z": 0.35,
  "latitude": 11.0168,
  "longitude": 76.9558,
  "speed": 2.4,
  "tamper": 0,
  "sos": 0
}
```

---

# 6. Sequence Processing

The AI models are temporal models.

A single sensor reading should not be passed directly to the complete pipeline.

The current sequence length is:

```text
30 readings
```

The backend must therefore maintain a rolling window of 30 consecutive readings.

```text
Reading 1
Reading 2
Reading 3
...
Reading 28
Reading 29
Reading 30
       |
       v
 AI Prediction
```

For continuous monitoring, a rolling window can be used:

```text
Window 1

1  2  3  4  ... 28 29 30
                     |
                     v
                  Prediction


Window 2

2  3  4  5  ... 29 30 31
                     |
                     v
                  Prediction


Window 3

3  4  5  6  ... 30 31 32
                     |
                     v
                  Prediction
```

The backend can decide how frequently predictions should be generated.

---

## Insufficient Data

If fewer than 30 readings are available, the temporal AI pipeline should not be executed.

Example response:

```json
{
  "status": "waiting",
  "message": "30 readings are required for prediction",
  "received_readings": 18,
  "required_readings": 30
}
```

---

# 7. Feature Engineering

The AI pipeline derives additional features from the raw sensor data.

## 7.1 Acceleration Magnitude

Calculated using:

```text
accel_x
accel_y
accel_z
```

Conceptually:

```text
acceleration_magnitude =
sqrt(accel_x² + accel_y² + accel_z²)
```

---

## 7.2 Gyroscope Magnitude

Calculated using:

```text
gyro_x
gyro_y
gyro_z
```

Conceptually:

```text
gyroscope_magnitude =
sqrt(gyro_x² + gyro_y² + gyro_z²)
```

---

## 7.3 Heart Rate Change

A temporal feature calculated from consecutive heart-rate readings:

```text
heart_rate_change
```

---

## 7.4 Speed Change

A temporal feature calculated from consecutive speed readings:

```text
speed_change
```

---

## Important

The backend should provide the raw sensor fields.

The ML preprocessing/inference layer should perform the required feature engineering.

Do not independently recreate preprocessing in the React Native application.

---

# 8. Model Outputs

The complete pipeline produces:

```text
Activity
Behavior
Anomaly
Risk
```

Example:

```json
{
  "activity": "walking",
  "behavior": "walking",
  "anomaly_detected": false,
  "anomaly_score": 0.12,
  "risk_level": "LOW"
}
```

If model confidence values are exposed:

```json
{
  "activity": "struggle",
  "activity_confidence": 0.97,
  "behavior": "struggle",
  "behavior_confidence": 0.94,
  "anomaly_detected": true,
  "anomaly_score": 0.31,
  "risk_level": "CRITICAL",
  "risk_confidence": 0.96
}
```

The final API response structure can be adjusted by the backend developer.

---

# 9. Project Structure

```text
Silambumodel/
│
├── data/
│   ├── raw/
│   │   └── silambu_sensor_data.csv
│   │
│   └── processed/
│       ├── silambu_processed.csv
│       └── behavior_label_mapping.json
│
├── models/
│   ├── activity_model.keras
│   ├── activity_label_encoder.pkl
│   │
│   ├── behavior_model.keras
│   ├── behavior_scaler.pkl
│   │
│   ├── anomaly_detector.pkl
│   ├── anomaly_scaler.pkl
│   ├── anomaly_metadata.json
│   │
│   ├── risk_model.pkl
│   └── risk_metadata.json
│
├── src/
│   ├── preprocessing.py
│   ├── activity_model.py
│   ├── behavior_model.py
│   ├── anomaly_detector.py
│   ├── risk_model.py
│   └── inference_pipeline.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 10. Requirements

## Python

Recommended Python version:

```text
Python 3.11
```

## Main Dependencies

The project uses:

```text
numpy
pandas
scikit-learn
xgboost
tensorflow
joblib
```

The complete dependency versions are defined in:

```text
requirements.txt
```

The Docker image installs these dependencies automatically.

---

# 11. Installation

## Clone Repository

```bash
git clone https://github.com/Dharavikasini/Silambumodel.git
cd Silambumodel
```

## Create Virtual Environment

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 12. Running the Existing Pipeline

The current inference pipeline can be executed using:

```bash
python src/inference_pipeline.py
```

The development version provides a runtime testing interface.

It allows selection of:

1. Scenario
2. Starting window index

Available scenarios:

```text
1. normal_rest
2. normal_walking
3. normal_running
4. sudden_movement
5. possible_struggle
6. unusual_location
7. tampering
8. emergency_sos
```

The selected window contains 30 readings.

---

# 13. Docker

Docker is provided to make the ML environment reproducible.

The Docker container includes:

- Python 3.11
- ML dependencies
- Source code
- Trained models
- Required data files

The local `.venv` is **not copied into Docker**.

Docker creates an isolated environment from the `Dockerfile` and `requirements.txt`.

---

## Build Docker Image

From the project root:

```bash
docker build -t silambu-ai .
```

---

## Run Docker Container

```bash
docker run -it --rm silambu-ai
```

The current development container runs:

```text
src/inference_pipeline.py
```

and provides the runtime testing interface.

---

## Verify Docker Image

```bash
docker images
```

Expected:

```text
REPOSITORY    TAG       IMAGE ID
silambu-ai    latest    ...
```

---

## Docker Architecture

```text
Docker Image
│
├── Python 3.11
├── requirements.txt
├── src/
├── models/
├── data/
└── inference pipeline
       |
       v
   Prediction
```

---

## Important Docker Note

The current Docker setup is intended to containerize the ML/inference environment.

For final application deployment, the AI module can be integrated into the Python backend container or exposed as a dedicated internal ML service.

---

# 14. Backend Integration

The AI repository is intended to be integrated into the Python backend.

Recommended backend architecture:

```text
backend/
│
├── app/
│   ├── api/
│   │   └── routes/
│   │
│   ├── services/
│   │   ├── sensor_service.py
│   │   ├── prediction_service.py
│   │   └── alert_service.py
│   │
│   ├── database/
│   │   ├── models.py
│   │   └── connection.py
│   │
│   └── ml/
│       └── silambu_model/
│
└── main.py
```

The exact backend structure can be changed.

The important separation is:

```text
API
 |
 +-- Sensor Service
 |
 +-- Database Service
 |
 +-- AI Prediction Service
 |
 +-- Alert Service
```

---

# 15. API Contract

The backend should expose an endpoint for sensor data.

Example:

```text
POST /api/sensor-data
```

Example request:

```json
{
  "child_id": "CHILD_001",
  "timestamp": "2026-09-19T12:30:00",

  "heart_rate": 112,
  "spo2": 97,

  "accel_x": 2.31,
  "accel_y": 1.12,
  "accel_z": 9.81,

  "gyro_x": 0.41,
  "gyro_y": 1.22,
  "gyro_z": 0.35,

  "latitude": 11.0168,
  "longitude": 76.9558,

  "speed": 2.4,

  "tamper": 0,
  "sos": 0
}
```

The backend should process the request as:

```text
Receive Reading
      |
      v
Validate Reading
      |
      v
Store Reading
      |
      v
Update 30-Reading Window
      |
      v
Run AI Pipeline
      |
      v
Store Prediction
      |
      v
Return Response
```

---

# 16. FastAPI Integration Example

The final backend can use FastAPI or another Python API framework.

Example:

```python
from fastapi import FastAPI

app = FastAPI()


@app.post("/api/sensor-data")
def receive_sensor_data(data):
    # Validate sensor data
    # Store sensor reading
    # Update 30-reading window
    # Run Silambu AI
    # Store prediction
    # Return prediction

    return {
        "status": "success"
    }
```

The model logic should preferably be placed inside a dedicated service rather than directly inside the API route.

Example:

```python
@app.post("/api/sensor-data")
def receive_sensor_data(data):

    prediction = prediction_service.process(data)

    return prediction
```

---

# 17. PostgreSQL Integration

The AI module does not need to directly manage PostgreSQL.

Recommended architecture:

```text
PostgreSQL
     ^
     |
Python Backend
     ^
     |
Silambu AI
```

The Python backend should manage:

- Database connections
- Sensor data insertion
- Prediction insertion
- Historical queries
- Child/device relationships

---

## Suggested Sensor Table

Table:

```text
sensor_readings
```

Suggested columns:

```text
id
child_id
timestamp

heart_rate
spo2

accel_x
accel_y
accel_z

gyro_x
gyro_y
gyro_z

latitude
longitude
speed

tamper
sos
```

---

## Suggested Prediction Table

Table:

```text
ai_predictions
```

Suggested columns:

```text
id
child_id
timestamp

activity
activity_confidence

behavior
behavior_confidence

anomaly_detected
anomaly_score

risk_level
risk_confidence
```

The final database schema can be modified according to the backend architecture.

---

# 18. React Native Integration

React Native should communicate with the Python backend.

It should **not directly load or execute the Python ML models**.

Recommended architecture:

```text
React Native
      |
      | HTTP / REST API
      v
Python Backend
      |
      v
Silambu AI
      |
      v
PostgreSQL
```

React Native receives the prediction result from the backend.

Example:

```json
{
  "activity": "walking",
  "behavior": "walking",
  "anomaly_detected": false,
  "anomaly_score": 0.12,
  "risk_level": "LOW"
}
```

The mobile application can display:

```text
Current Activity
Current Behavior
Risk Level
Anomaly Status
Location
Emergency Status
```

---

# 19. Prediction Flow

Complete application flow:

```text
                Wearable
                   |
                   v
            Sensor Readings
                   |
                   v
            Python Backend
                   |
                   v
          Input Validation
                   |
                   v
             PostgreSQL
                   |
                   v
       30-Reading Rolling Window
                   |
                   v
          Feature Engineering
                   |
          +--------+--------+
          |        |        |
          v        v        v
      Activity  Behavior  Anomaly
          |        |        |
          +--------+--------+
                   |
                   v
             Risk Model
                   |
                   v
             Final Result
                   |
          +--------+--------+
          |                 |
          v                 v
      PostgreSQL       React Native
```

---

# 20. Model Files

All trained models are stored inside:

```text
models/
```

## Activity Model

```text
activity_model.keras
activity_label_encoder.pkl
```

The encoder is required to convert the model's numeric output into the corresponding activity label.

---

## Behavior Model

```text
behavior_model.keras
behavior_scaler.pkl
```

The behavior label mapping is stored in:

```text
data/processed/behavior_label_mapping.json
```

---

## Anomaly Detector

```text
anomaly_detector.pkl
anomaly_scaler.pkl
anomaly_metadata.json
```

---

## Risk Model

```text
risk_model.pkl
risk_metadata.json
```

---

# 21. Important Integration Rules

## Rule 1 — Maintain the 30-reading window

The current temporal models expect:

```text
30 consecutive readings
```

Do not change the sequence length without retraining and validating the affected models.

---

## Rule 2 — Preserve feature order

The activity model expects:

```text
heart_rate
spo2
accel_x
accel_y
accel_z
gyro_x
gyro_y
gyro_z
speed
acceleration_magnitude
gyroscope_magnitude
```

The behavior model expects:

```text
heart_rate
spo2
accel_x
accel_y
accel_z
gyro_x
gyro_y
gyro_z
speed
acceleration_magnitude
gyroscope_magnitude
heart_rate_change
speed_change
```

Do not change the feature order.

---

## Rule 3 — Preserve preprocessing

The models depend on the preprocessing and scaling used during training.

Do not introduce a different normalization or scaling method in the backend.

Use the preprocessing logic provided by this repository.

---

## Rule 4 — Preserve label mappings

Do not manually change activity or behavior label mappings.

The activity label encoder is:

```text
models/activity_label_encoder.pkl
```

The behavior mapping is:

```text
data/processed/behavior_label_mapping.json
```

---

## Rule 5 — Do not load models per request

Models should be loaded once when the backend starts.

Recommended:

```text
Backend Startup
      |
      v
Load Models
      |
      v
Keep Models in Memory
      |
      v
API Requests
      |
      v
Prediction
```

Avoid:

```text
API Request
     |
     v
Load Model
     |
     v
Prediction
     |
     v
Unload Model
```

---

## Rule 6 — Handle insufficient data

If fewer than 30 readings are available, do not run the complete temporal inference pipeline.

---

## Rule 7 — Keep ML logic separate

Do not put the complete model implementation directly inside API route functions.

Prefer:

```text
API Route
    |
    v
Prediction Service
    |
    v
Silambu AI Pipeline
```

---

## Rule 8 — Do not modify trained model files

Do not manually edit:

```text
*.keras
*.pkl
```

These are trained artifacts.

---

# 22. Testing

## Run Local Pipeline

```bash
python src/inference_pipeline.py
```

Select a scenario and starting index.

Example:

```text
Scenario number: 5
Starting window index: 0
```

This selects:

```text
possible_struggle
```

and processes a 30-reading window.

---

## Run Docker Version

```bash
docker build -t silambu-ai .
```

Then:

```bash
docker run -it --rm silambu-ai
```

---

## Test Scenarios

The current development dataset contains:

```text
normal_rest
normal_walking
normal_running
sudden_movement
possible_struggle
unusual_location
tampering
emergency_sos
```

Recommended development tests:

| Scenario | Risk Mapping |
|---|---|
| normal_rest | LOW |
| normal_walking | LOW |
| normal_running | LOW |
| unusual_location | MEDIUM |
| sudden_movement | HIGH |
| tampering | HIGH |
| possible_struggle | CRITICAL |
| emergency_sos | CRITICAL |

These mappings represent the current development configuration.

---

# 23. Model Performance

The current models were evaluated using the available development dataset.

## Activity Recognition

Current test accuracy:

```text
99.51%
```

Test samples:

```text
9,995
```

---

## Risk Prediction

Current test accuracy:

```text
100%
```

---

## Important Performance Note

The current dataset is a **synthetic development dataset**.

Therefore:

```text
Performance on synthetic data
            !=
Real-world performance
```

The current accuracy values should not be interpreted as guaranteed production performance.

Real-world validation is required using data collected from the actual wearable hardware.

---

# 24. Current Status

## ML Development

```text
[x] Dataset preparation
[x] Data preprocessing
[x] Feature engineering
[x] Activity recognition model
[x] Behavior analysis model
[x] Anomaly detection model
[x] Risk prediction model
[x] Model serialization
[x] End-to-end inference pipeline
[x] Runtime scenario testing
[x] Docker environment
```

## Application Integration

```text
[ ] Python backend API
[ ] PostgreSQL integration
[ ] React Native integration
[ ] Wearable sensor integration
[ ] Real-world data collection
[ ] Real-world model validation
[ ] Production deployment
[ ] Monitoring and logging
[ ] Alert/notification integration
```

---

# 25. Future Improvements

Potential future improvements include:

- Real-world sensor data collection
- Personalized child behavior baselines
- Improved anomaly detection
- Reduction of false positives
- Continuous model evaluation
- Model versioning
- Inference latency optimization
- Production monitoring
- API authentication
- Model performance monitoring
- Real-world validation
- Better alert prioritization

---

# 26. Developer Handoff

This section summarizes what a developer integrating this repository needs to know.

## Input

The AI pipeline requires:

```text
30 consecutive sensor readings
```

Each reading should contain:

```text
timestamp
heart_rate
spo2
accel_x
accel_y
accel_z
gyro_x
gyro_y
gyro_z
latitude
longitude
speed
tamper
sos
```

---

## Processing

The AI pipeline performs:

```text
Raw Sensor Data
       |
       v
Feature Engineering
       |
       v
Activity Recognition
       |
       v
Behavior Analysis
       |
       v
Anomaly Detection
       |
       v
Risk Prediction
```

---

## Output

The backend can expose:

```text
activity
behavior
anomaly_detected
anomaly_score
risk_level
risk_confidence
```

---

## Backend Responsibility

The backend developer should implement:

```text
API
Sensor ingestion
Input validation
30-reading rolling window
AI model invocation
PostgreSQL storage
Prediction storage
Authentication
Alert handling
React Native communication
```

---

## React Native Responsibility

The React Native developer should implement:

```text
Mobile UI
Dashboard
Activity display
Behavior display
Risk display
Location display
Alerts
API communication
```

---

## ML Responsibility

The ML module is responsible for:

```text
Preprocessing
Feature engineering
Model loading
Activity prediction
Behavior prediction
Anomaly detection
Risk prediction
```

---

# 27. Recommended Integration Boundary

The recommended integration boundary is:

```text
                 PYTHON BACKEND
                       |
                       v
              Prediction Service
                       |
                       v
              Silambu AI Module
                       |
          +------------+------------+
          |            |            |
          v            v            v
      Activity      Behavior     Anomaly
          |            |            |
          +------------+------------+
                       |
                       v
                  Risk Model
                       |
                       v
                  Prediction
```

The backend should interact with the ML module through a clear service interface.

Conceptually:

```python
prediction = silambu_ai.predict(sensor_window)
```

Input:

```text
30 sensor readings
```

Output:

```text
activity
behavior
anomaly
risk
```

This keeps the ML implementation independent from API and database logic.

---

# Production Considerations

Before production deployment, the following should be implemented.

## Input Validation

Validate:

- Missing values
- Invalid timestamps
- Invalid GPS values
- Invalid sensor values
- Missing sensor readings
- Invalid SOS/tamper values

---

## Error Handling

The backend should handle:

- Invalid sensor data
- Insufficient readings
- Model loading failures
- Prediction exceptions
- Database failures
- API failures

---

## Logging

Recommended logging fields:

```text
timestamp
child/device ID
model version
prediction result
processing time
error information
```

Avoid logging unnecessary sensitive information.

---

# Security

Do not store credentials inside the ML repository.

Never commit:

```text
.env
API keys
Database passwords
JWT secrets
Private keys
Cloud credentials
```

Use environment variables in the backend.

---

# Model Versioning

When models are retrained, maintain model versions.

Example:

```text
models/
    activity_model_v1.keras
    behavior_model_v1.keras
    anomaly_model_v1.pkl
    risk_model_v1.pkl
```

A model metadata file can also contain:

```json
{
  "model_version": "1.0.0",
  "trained_date": "2026-09-19"
}
```

Production models should not be replaced without validation.

---

# Quick Start for Backend Developer

```text
1. Clone the repository.

2. Create a Python 3.11 environment.

3. Install requirements.

4. Run the existing inference pipeline.

5. Verify that the trained models load correctly.

6. Build the Docker image.

7. Run the Docker container.

8. Integrate the AI module into the Python backend.

9. Implement the sensor-data API.

10. Maintain the 30-reading rolling window.

11. Pass the window to the AI prediction service.

12. Store sensor readings in PostgreSQL.

13. Store AI predictions in PostgreSQL.

14. Return predictions through the backend API.

15. Connect React Native to the backend.
```

---

# Final Integration Contract

The AI module should be treated as a prediction service.

## INPUT

```text
30 consecutive sensor readings
```

Each reading:

```text
timestamp
heart_rate
spo2
accel_x
accel_y
accel_z
gyro_x
gyro_y
gyro_z
latitude
longitude
speed
tamper
sos
```

## PROCESSING

```text
Sensor Data
     |
     v
30-Reading Window
     |
     v
Feature Engineering
     |
     +------------------+
     |                  |
     v                  v
Activity            Behavior
DeepConvLSTM           LSTM
     |                  |
     +--------+---------+
              |
              v
          Anomaly
      Isolation Forest
              |
              v
            Risk
          XGBoost
              |
              v
         Final Result
```

## OUTPUT

```text
activity
behavior
anomaly_detected
anomaly_score
risk_level
risk_confidence
```

---

# Important Integration Rule

The trained models depend on the preprocessing pipeline, feature definitions, feature order, sequence length, scalers, and label mappings used during training.

Therefore, the backend integration should reuse the existing inference and preprocessing logic wherever possible.

Do not independently recreate the preprocessing logic in the React Native application or backend unless it is verified against the existing ML implementation.

Changes to any of the following should be coordinated with the ML developer:

```text
Feature order
Feature calculation
Scaling
Sequence length
Label mappings
Model input shape
Model files
```

---

# Disclaimer

Silambu is currently a prototype/research system.

The AI predictions are intended to support safety monitoring and should not be treated as guaranteed detection of emergencies, threats, or unsafe situations.

The current model evaluation is based on a synthetic development dataset. Real-world validation using actual wearable sensor data is required before production deployment.