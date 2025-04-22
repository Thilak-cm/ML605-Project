# 🚕 NYC Taxi Demand Predictor

A real-time, containerized ML system for predicting hourly taxi demand in New York City. Built with a modular, production-ready architecture using machine learning, live weather/traffic data, and REST APIs.

---

## 🔍 Overview

This project predicts short-term (30–60 minute) taxi demand across NYC zones by combining historical taxi trip data with real-time weather and traffic inputs. It is built with scalability in mind using containerized services, model versioning, and modular microservices.

---

## 🧱 System Architecture

![System Flow Diagram](./taxi%20demand%20project%20flow%20chart.png)

### Key Modules:
- **Taxi & Weather Data** (historical): Feature engineering + preprocessing
- **Taxi Demand Model**: Trained ML model (XGBoost/Transformer)
- **Real-Time Data Integration**: Live weather & traffic inputs
- **Prediction Service (API)**: FastAPI serving the trained model
- **B2B Use Cases**: Dashboard monitoring, driver incentives, shift planning

---
