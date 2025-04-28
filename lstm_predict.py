import os
import pandas as pd
import numpy as np
import joblib
import logging
from tensorflow.keras.models import load_model
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Model parameters
    SEQUENCE_LENGTH = 24  # 24 hours of historical data
    PREDICTION_LENGTH = 24  # Predict next 24 hours
    RANDOM_SEED = 42
    
    # File paths
    DATA_PATH = 'data_from_2024/taxi_demand_dataset.csv'
    MODEL_DIR = 'models'
    MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_taxi_model.keras')
    SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.save')
    
    # Features
    FEATURES = [
        'temp', 'feels_like', 'wind_speed', 'rain_1h',
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_holiday', 'is_rush_hour',
        'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',  # Lag features
        'demand_rolling_mean_24h', 'demand_rolling_std_24h',  # Rolling statistics
        'demand_change_1h'  # Change features
    ]
    TARGET = 'demand'

def load_lstm_model(model_path: str = Config.MODEL_PATH, scaler_path: str = Config.SCALER_PATH) -> Tuple:
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
    
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    logger.info(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def load_sample_data(data_path: str = Config.DATA_PATH, sample_size: int = 100) -> pd.DataFrame:
    """
    Load sample data for prediction.
    
    Args:
        data_path: Path to the dataset
        sample_size: Number of samples to load
        
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading sample data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Take a random sample
    df = df.sample(n=sample_size, random_state=Config.RANDOM_SEED)
    
    # Convert hour to datetime if present
    if 'hour' in df.columns:
        df['timestamp'] = pd.to_datetime(df['hour'])
        df = df.sort_values('timestamp')
    
    logger.info(f"Loaded {len(df)} samples")
    return df

def prepare_features_from_dict(features_dict: Dict[str, List[float]], scaler) -> np.ndarray:
    """
    Prepare input data from a features dictionary for API integration.
    
    Args:
        features_dict: Dictionary containing feature arrays
        scaler: Fitted scaler
        
    Returns:
        Scaled and reshaped input for LSTM model
    """
    # Check if we have all required features
    missing_features = set(Config.FEATURES) - set(features_dict.keys())
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Ensure we have at least SEQUENCE_LENGTH data points
    for feature, values in features_dict.items():
        if len(values) < Config.SEQUENCE_LENGTH:
            raise ValueError(f"Feature {feature} has {len(values)} points, but {Config.SEQUENCE_LENGTH} are required")
    
    # Create a DataFrame from the features dictionary
    df = pd.DataFrame(features_dict)
    
    # Add dummy target column if not present
    if Config.TARGET not in df.columns:
        # Use a reasonable default value (e.g., mean of historical data or 0)
        df[Config.TARGET] = 0.0
    
    # Get the last SEQUENCE_LENGTH rows
    data = df.iloc[-Config.SEQUENCE_LENGTH:].copy()
    
    # Ensure all features are present
    all_required_columns = Config.FEATURES + [Config.TARGET]
    for col in all_required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column {col} not found in data")
    
    # Reorder columns to match training data
    data = data[all_required_columns]
    
    # Scale the data
    scaled_data = scaler.transform(data)
    
    # Reshape for LSTM input (samples, time steps, features)
    X = scaled_data.reshape(1, Config.SEQUENCE_LENGTH, len(all_required_columns))
    
    logger.info(f"Prepared input shape: {X.shape}")
    return X

def prepare_input_data(df: pd.DataFrame, scaler, sequence_length: int = Config.SEQUENCE_LENGTH) -> np.ndarray:
    """
    Prepare input data for prediction from DataFrame.
    
    Args:
        df: DataFrame with features
        scaler: Fitted scaler
        sequence_length: Length of input sequence
    
    Returns:
        Scaled input sequence
    """
    # Ensure all required features are present
    all_features = Config.FEATURES + [Config.TARGET]
    missing_features = set(all_features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Get the last sequence_length rows
    data = df[all_features].iloc[-sequence_length:].copy()
    
    # Scale the data
    scaled_data = scaler.transform(data)
    
    # Reshape for LSTM input (samples, time steps, features)
    X = scaled_data.reshape(1, sequence_length, len(all_features))
    
    logger.info(f"Prepared input shape: {X.shape}")
    return X

def predict_demand_from_features(
    features_dict: Dict[str, List[float]], 
    zone_id: Optional[int] = None,
    model_path: str = Config.MODEL_PATH, 
    scaler_path: str = Config.SCALER_PATH
) -> Dict[str, Union[List[float], int]]:
    """
    Make predictions using features dict (for API integration).
    
    Args:
        features_dict: Dictionary of feature arrays
        zone_id: Optional zone identifier
        model_path: Path to the model file
        scaler_path: Path to the scaler file
        
    Returns:
        Dictionary with predictions and metadata
    """
    try:
        # Load model and scaler
        model, scaler = load_lstm_model(model_path, scaler_path)
        
        # Prepare input data
        input_data = prepare_features_from_dict(features_dict, scaler)
        
        # Make prediction
        scaled_predictions = model.predict(input_data, verbose=0)
        
        # Inverse transform the predictions
        dummy = np.zeros((scaled_predictions.shape[1], scaler.n_features_in_))
        dummy[:, -1] = scaled_predictions[0]
        predictions = scaler.inverse_transform(dummy)[:, -1]
        
        # Format response
        result = {
            "demand": predictions.tolist(),
            "model": "lstm"
        }
        
        if zone_id is not None:
            result["zone_id"] = zone_id
            
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

def predict_demand(model, scaler, input_data: np.ndarray) -> np.ndarray:
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
    logger.info("Running LSTM prediction")
    scaled_predictions = model.predict(input_data, verbose=0)
    
    # Inverse transform the predictions
    # Create a dummy array with the same shape as the original data
    dummy = np.zeros((scaled_predictions.shape[1], scaler.n_features_in_))
    # Put the predictions in the last column (demand)
    dummy[:, -1] = scaled_predictions[0]
    # Inverse transform
    predictions = scaler.inverse_transform(dummy)[:, -1]
    
    logger.info(f"Generated {len(predictions)} predictions")
    return predictions

def predict_future_demand(
    df: pd.DataFrame,
    model_path: str = Config.MODEL_PATH, 
    scaler_path: str = Config.SCALER_PATH
) -> np.ndarray:
    """
    End-to-end prediction function for external use.
    
    Args:
        df: DataFrame with required features
        model_path: Path to the model file
        scaler_path: Path to the scaler file
        
    Returns:
        Array of predicted demand values
    """
    try:
        # Load model and scaler
        model, scaler = load_lstm_model(model_path, scaler_path)
        
        # Prepare input data
        input_data = prepare_input_data(df, scaler)
        
        # Make prediction
        predictions = predict_demand(model, scaler, input_data)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error predicting future demand: {str(e)}")
        raise

def main():
    """Run a sample prediction."""
    try:
        # Load the model and scaler
        model, scaler = load_lstm_model()
        logger.info("Successfully loaded model and scaler")
        
        # Load and prepare test data
        df = load_sample_data()
        
        # Prepare input data
        input_data = prepare_input_data(df, scaler)
        
        # Make predictions
        predictions = predict_demand(model, scaler, input_data)
        
        # Print results
        print("\nPredicted demand for next 24 hours:")
        for hour, pred in enumerate(predictions):
            print(f"Hour {hour+1}: {pred:.2f}")
            
        # Print actual values for comparison
        actual_values = df[Config.TARGET].iloc[-Config.PREDICTION_LENGTH:].values
        print("\nActual demand for last 24 hours:")
        for hour, actual in enumerate(actual_values):
            print(f"Hour {hour+1}: {actual:.2f}")
        
        # Demonstrate API-like usage
        print("\nDemonstration of API integration function:")
        # Extract features from DataFrame to dictionary
        features_dict = {
            feature: df[feature].iloc[-Config.SEQUENCE_LENGTH:].tolist() 
            for feature in Config.FEATURES
        }
        
        # Call the API integration function
        result = predict_demand_from_features(features_dict, zone_id=1)
        print(f"API result: {result}")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main() 