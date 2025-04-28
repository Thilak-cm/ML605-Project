import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_lstm_model(model_path: str = 'models/lstm_taxi_model.h5', scaler_path: str = 'models/scaler.save'):
    """
    Load the trained LSTM model and scaler.
    
    Args:
        model_path: Path to the saved model file
        scaler_path: Path to the saved scaler file
    
    Returns:
        Tuple of (model, scaler)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def prepare_input_data(df: pd.DataFrame, scaler, sequence_length: int = 24):
    """
    Prepare input data for prediction.
    
    Args:
        df: DataFrame with features
        scaler: Fitted scaler
        sequence_length: Length of input sequence
    
    Returns:
        Scaled input sequence
    """
    # Select features
    features = [
        'temp', 'feels_like', 'wind_speed', 'rain_1h',
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_holiday', 'is_rush_hour',
        'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',
        'demand_rolling_mean_24h', 'demand_rolling_std_24h',
        'demand_change_1h', 'demand'
    ]
    
    # Get the last sequence_length rows
    data = df[features].iloc[-sequence_length:].copy()
    
    # Scale the data
    scaled_data = scaler.transform(data)
    
    # Reshape for LSTM input (samples, time steps, features)
    X = scaled_data.reshape(1, sequence_length, len(features))
    
    return X

def predict_demand(model, scaler, input_data):
    """
    Make predictions using the LSTM model.
    
    Args:
        model: Loaded LSTM model
        scaler: Fitted scaler
        input_data: Prepared input data
    
    Returns:
        Predicted demand values
    """
    # Make prediction
    scaled_predictions = model.predict(input_data)
    
    # Inverse transform the predictions
    # Create a dummy array with the same shape as the original data
    dummy = np.zeros((scaled_predictions.shape[1], scaler.n_features_in_))
    # Put the predictions in the last column (demand)
    dummy[:, -1] = scaled_predictions[0]
    # Inverse transform
    predictions = scaler.inverse_transform(dummy)[:, -1]
    
    return predictions

def main():
    # Load the model and scaler
    try:
        model, scaler = load_lstm_model()
        logger.info("Successfully loaded model and scaler")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
    
    # Load and prepare test data
    try:
        df = pd.read_csv('data_from_2024/taxi_demand_dataset.csv')
        # Take a random sample of 100 rows
        df = df.sample(n=100, random_state=42)
        logger.info("Successfully loaded test data")
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        return
    
    # Prepare input data
    try:
        input_data = prepare_input_data(df, scaler)
        logger.info("Successfully prepared input data")
    except Exception as e:
        logger.error(f"Error preparing input data: {str(e)}")
        return
    
    # Make predictions
    try:
        predictions = predict_demand(model, scaler, input_data)
        logger.info("Successfully made predictions")
        
        # Print results
        print("\nPredicted demand for next 24 hours:")
        for hour, pred in enumerate(predictions):
            print(f"Hour {hour+1}: {pred:.2f}")
            
        # Print actual values for comparison
        actual_values = df['demand'].iloc[-24:].values
        print("\nActual demand for last 24 hours:")
        for hour, actual in enumerate(actual_values):
            print(f"Hour {hour+1}: {actual:.2f}")
            
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return

if __name__ == "__main__":
    main() 