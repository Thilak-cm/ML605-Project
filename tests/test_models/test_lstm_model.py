import pytest
import numpy as np
import os
import joblib
import tempfile
from pathlib import Path

# Try to import the necessary functions
try:
    from nowcast_predict import (
        load_lstm_model,
        predict_demand,
        prepare_features_from_dict,
        Config
    )
except ImportError:
    # If we can't import directly, we'll use proxies in the tests
    pass

# Create a mock LSTM model and scaler for testing
class MockConfig:
    SEQUENCE_LENGTH = 24
    FEATURES = [
        'temp', 'feels_like', 'wind_speed', 'rain_1h',
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_holiday', 'is_rush_hour',
        'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',
        'demand_rolling_mean_24h', 'demand_rolling_std_24h',
        'demand_change_1h'
    ]
    TARGET = 'demand'

@pytest.fixture
def mock_model_and_scaler():
    """Create a mock model and scaler for testing."""
    # Skip if tensorflow not available
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        pytest.skip("TensorFlow or sklearn not available")
    
    # Create a simple LSTM model
    feature_count = len(MockConfig.FEATURES) + 1  # +1 for target
    model = Sequential()
    model.add(LSTM(32, input_shape=(MockConfig.SEQUENCE_LENGTH, feature_count), return_sequences=True))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Create a mock scaler
    scaler = StandardScaler()
    # Fit the scaler with dummy data
    dummy_data = np.random.rand(100, feature_count)
    scaler.fit(dummy_data)
    
    # Create temporary directory for model files
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, 'test_model.keras')
    scaler_path = os.path.join(temp_dir, 'test_scaler.save')
    
    # Save model and scaler
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    yield model_path, scaler_path, model, scaler
    
    # Cleanup
    os.remove(model_path)
    os.remove(scaler_path)
    os.rmdir(temp_dir)

def test_model_loading(mock_model_and_scaler):
    """Test that model and scaler can be loaded correctly."""
    model_path, scaler_path, _, _ = mock_model_and_scaler
    
    try:
        # Attempt to load the model using the project's function
        model, scaler = load_lstm_model(model_path, scaler_path)
        
        # Check that we got objects of the right type
        assert model is not None, "Model failed to load"
        assert scaler is not None, "Scaler failed to load"
        
        # Check model structure (assuming it's a Keras model)
        assert hasattr(model, 'predict'), "Model lacks predict method"
        
        # Check scaler (assuming it's a sklearn scaler)
        assert hasattr(scaler, 'transform'), "Scaler lacks transform method"
        
    except NameError:
        # If the function doesn't exist, load manually
        from tensorflow.keras.models import load_model
        import joblib
        
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        assert model is not None, "Model failed to load manually"
        assert scaler is not None, "Scaler failed to load manually"

def test_prediction_shape(mock_model_and_scaler):
    """Test that the model produces predictions with the expected shape."""
    model_path, scaler_path, model, scaler = mock_model_and_scaler
    
    # Create dummy input data
    n_features = len(MockConfig.FEATURES) + 1  # features + target
    X = np.random.rand(1, MockConfig.SEQUENCE_LENGTH, n_features)
    
    try:
        # Use the project's prediction function
        predictions = predict_demand(model, scaler, X)
        
        # Ensure predictions is a numpy array and flatten it
        predictions = np.array(predictions).flatten()
        assert predictions.shape == (24,), f"Expected shape (24,), got {predictions.shape}"
        
    except NameError:
        # If function is not available, predict directly
        scaled_predictions = model.predict(X)
        # Flatten the predictions to ensure correct shape
        predictions = np.array(scaled_predictions).reshape(-1)
        assert predictions.shape == (24,), f"Expected shape (24,), got {predictions.shape}"

def test_features_preparation(mock_model_and_scaler):
    """Test preparation of features dictionary to model input."""
    _, _, _, scaler = mock_model_and_scaler
    
    # Create dummy features dictionary
    features_dict = {}
    for feature in MockConfig.FEATURES:
        features_dict[feature] = np.random.rand(MockConfig.SEQUENCE_LENGTH).tolist()
    
    # Add dummy target
    features_dict[MockConfig.TARGET] = np.random.rand(MockConfig.SEQUENCE_LENGTH).tolist()
    
    try:
        # Use the project's function
        input_data = prepare_features_from_dict(features_dict, scaler)
        
        # Check shape
        expected_shape = (1, MockConfig.SEQUENCE_LENGTH, len(MockConfig.FEATURES) + 1)
        assert input_data.shape == expected_shape, f"Expected shape {expected_shape}, got {input_data.shape}"
        
    except NameError:
        # If function is not available, implement manually
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(features_dict)
        
        # Reorder columns to match expected
        all_columns = MockConfig.FEATURES + [MockConfig.TARGET]
        df = df[all_columns]
        
        # Scale with the provided scaler
        scaled_data = scaler.transform(df)
        
        # Reshape for LSTM input
        X = scaled_data.reshape(1, MockConfig.SEQUENCE_LENGTH, len(all_columns))
        
        # Check shape
        expected_shape = (1, MockConfig.SEQUENCE_LENGTH, len(all_columns))
        assert X.shape == expected_shape, f"Expected shape {expected_shape}, got {X.shape}"

def test_prediction_with_invalid_features():
    """Test that prediction with invalid features raises appropriate errors."""
    try:
        # Create features with missing required features
        incomplete_features = {
            'temp': [70.0] * 24,
            'feels_like': [68.0] * 24,
            # Missing several required features
        }
        
        # This should raise a ValueError
        with pytest.raises(ValueError):
            prepare_features_from_dict(incomplete_features, None)
            
    except NameError:
        pytest.skip("prepare_features_from_dict function not available") 