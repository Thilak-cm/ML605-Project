import pandas as pd
import numpy as np
import joblib
import holidays
from datetime import datetime
import os
from typing import Dict

def load_model(model_path: str = 'time_based_model.joblib') -> Dict:
    """
    Load the trained model and scaler from disk.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Dictionary containing model, scaler, and feature names
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    return joblib.load(model_path)

def predict_future_demand(timestamp: pd.Timestamp, zone_id: int, model_path: str = 'time_based_model.joblib') -> float:
    """
    Predict demand for a future timestamp and zone using only time-based features.
    
    Args:
        timestamp: Future timestamp to predict for
        zone_id: Zone ID to predict for
        model_path: Path to the saved model
        
    Returns:
        Predicted demand
    """
    model_data = load_model(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Create time-based features
    features = {
        'hour_of_day': timestamp.hour,
        'day_of_week': timestamp.dayofweek,
        'month': timestamp.month,
        'hour_sin': np.sin(2 * np.pi * timestamp.hour/24),
        'hour_cos': np.cos(2 * np.pi * timestamp.hour/24),
        'day_sin': np.sin(2 * np.pi * timestamp.dayofweek/7),
        'day_cos': np.cos(2 * np.pi * timestamp.dayofweek/7),
        'is_weekend': int(timestamp.dayofweek >= 5),
        'is_holiday': int(timestamp.date() in holidays.US()),
        'is_rush_hour': int(timestamp.hour in [7, 8, 9, 16, 17, 18]),
        'zone_id': zone_id  # Use zone_id directly as numeric feature
    }
    
    # Create DataFrame and ensure correct feature order
    input_df = pd.DataFrame([features])
    input_df = input_df[feature_names]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    return float(prediction)

if __name__ == "__main__":
    # Example usage
    future_date = pd.Timestamp('2025-05-22 18:00:00')
    
    # Sample a zone value from the dataset
    df = pd.read_csv('data_from_2024/merged_features.csv')
    # zone_id = df['zone_id'].sample(n=1).iloc[0]
    zone_id = 233
    
    prediction = predict_future_demand(future_date, zone_id)
    print(f"\nPrediction for Zone {zone_id} on {future_date}: {prediction:.2f}") 