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
- [13. Backend Integration](#13-backend-integration)
- [14. API Contract](#14-api-contract)
- [15. FastAPI Integration Example](#15-fastapi-integration-example)
- [16. PostgreSQL Integration](#16-postgresql-integration)
- [17. React Native Integration](#17-react-native-integration)
- [18. Prediction Flow](#18-prediction-flow)
- [19. Model Files](#19-model-files)
- [20. Important Integration Rules](#20-important-integration-rules)
- [21. Testing](#21-testing)
- [22. Model Performance](#22-model-performance)
- [23. Current Status](#23-current-status)
- [24. Future Improvements](#24-future-improvements)
- [25. Disclaimer](#25-disclaimer)

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
 DeepConvLSTM                LSTM
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