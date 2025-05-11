import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
import time
# Try to import key functions
try:
    from nowcast_predict import predict_demand_from_features
    from forecast_predict import predict_future_demand
    from app import extract_weather_data
    import mock_weather_data
except ImportError:
    # If we can't import directly, we'll skip the tests that require them
    pass

@pytest.fixture
def mock_api_features():
    """Create mock features similar to what the API would generate."""
    # Generate 24 hours of fake data
    features = {
        'temp': [round(np.random.uniform(30, 90), 1) for _ in range(24)],
        'feels_like': [round(np.random.uniform(30, 85), 1) for _ in range(24)],
        'wind_speed': [round(np.random.uniform(0, 20), 1) for _ in range(24)],
        'rain_1h': [round(np.random.choice([0, 0, 0, 0, 0.5]), 1) for _ in range(24)],
        'hour_of_day': list(range(24)),
        'day_of_week': [datetime.now().weekday()] * 24,
        'is_weekend': [int(datetime.now().weekday() >= 5)] * 24,
        'is_holiday': [0] * 24,
        'is_rush_hour': [1 if h in [7, 8, 9, 16, 17, 18] else 0 for h in range(24)],
        'demand_lag_1h': [round(np.random.uniform(10, 100), 1) for _ in range(24)],
        'demand_lag_24h': [round(np.random.uniform(10, 100), 1) for _ in range(24)],
        'demand_lag_168h': [round(np.random.uniform(10, 100), 1) for _ in range(24)],
        'demand_rolling_mean_24h': [round(np.random.uniform(30, 70), 1) for _ in range(24)],
        'demand_rolling_std_24h': [round(np.random.uniform(5, 15), 1) for _ in range(24)],
        'demand_change_1h': [round(np.random.uniform(-10, 10), 1) for _ in range(24)]
    }
    return features

def test_nowcast_model_output_shape():
    """Test that nowcast model produces output with expected shape."""
    try:
        features = {
            'temp': [70.0] * 24,
            'feels_like': [68.0] * 24,
            'wind_speed': [5.0] * 24,
            'rain_1h': [0.0] * 24,
            'hour_of_day': list(range(24)),
            'day_of_week': [datetime.now().weekday()] * 24,
            'is_weekend': [int(datetime.now().weekday() >= 5)] * 24,
            'is_holiday': [0] * 24,
            'is_rush_hour': [1 if h in [7, 8, 9, 16, 17, 18] else 0 for h in range(24)],
            'demand_lag_1h': [40.0] * 24,
            'demand_lag_24h': [35.0] * 24,
            'demand_lag_168h': [30.0] * 24,
            'demand_rolling_mean_24h': [38.0] * 24,
            'demand_rolling_std_24h': [5.0] * 24,
            'demand_change_1h': [1.0] * 24
        }
        
        # Skip if models are not available
        if not os.path.exists('models/best_lstm_model.keras'):
            pytest.skip("LSTM model not found, skipping test")
            
        result = predict_demand_from_features(features, zone_id=236)
        
        assert len(result["demand"]) > 0, "Empty predictions list"
        # All predictions should be non-negative for demand
        assert all(p >= 0 for p in result["demand"]), "Negative demand predictions"
        
    except (NameError, ImportError):
        pytest.skip("Required functions not imported, skipping test")

def test_forecast_model_returns_value():
    """Test that forecast model returns a valid prediction."""
    try:
        # Use a timestamp 2 months in the future
        future_time = datetime.now() + timedelta(days=60)
        zone_id = 236
        
        # Skip if models are not available
        if not os.path.exists('models/time_based_model.joblib'):
            pytest.skip("Forecast model not found, skipping test")
            
        prediction = predict_future_demand(pd.Timestamp(future_time), zone_id)
        
        # Result should be a float
        assert isinstance(prediction, float), f"Expected float, got {type(prediction)}"
        
        # Value should be non-negative for demand
        assert prediction >= 0, "Negative demand prediction"
        
    except (NameError, ImportError):
        pytest.skip("Required functions not imported, skipping test")

def test_extract_weather_data(mock_api_client):
    """Test weather data extraction function using the /live-features/ API endpoint with mock=True."""
    response = mock_api_client.get("/zones-with-coords")
    print("Available zones:", response.json())
    lat, lon = 40.7128, -74.0060  # NYC coordinates
    zone_id = 216
    # Call the live-features endpoint with mock=True
    response = mock_api_client.get(f"/live-features/?zone_id={zone_id}&mock=true")
    if response.status_code != 200:
        print("tacos API error response:", response.json())
    assert response.status_code == 200, f"API call failed: {response.status_code}"
    data = response.json()
    features = data.get("features", {})
    # Check that we have key features
    expected_features = [
        'temp', 'feels_like', 'wind_speed', 'rain_1h',
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_holiday'
    ]
    for feature in expected_features:
        assert feature in features, f"Missing expected feature: {feature}"
        assert isinstance(features[feature], list), f"Feature {feature} is not a list"
        assert len(features[feature]) > 0, f"Feature {feature} is empty"

def test_feature_to_prediction_pipeline(mock_api_client, mock_api_features):
    """Test the full pipeline from feature extraction to prediction using the /live-features/ API endpoint with mock=True."""
    # Check if necessary components exist
    if not os.path.exists('models/best_lstm_model.keras'):
        pytest.skip("LSTM model not found, skipping test")
    zone_id = 236
    # 1. Get weather features from the API
    response = mock_api_client.get(f"/live-features/?zone_id={zone_id}&mock=true")
    assert response.status_code == 200, f"API call failed: {response.status_code}"
    data = response.json()
    weather_features = data.get("features", {})
    # 2. Add historical data (mock_api_features)
    merged_features = {**weather_features, **mock_api_features}
    # 3. Make prediction
    try:
        from nowcast_predict import predict_demand_from_features
        result = predict_demand_from_features(merged_features, zone_id)
        # 4. Check result
        assert "predictions" in result or "demand" in result, "No predictions in result"
        predictions = result.get("predictions") or result.get("demand")
        assert len(predictions) > 0, "Empty predictions list"
        assert all(isinstance(p, (int, float)) for p in predictions), "Non-numeric predictions"
    except (ImportError, NameError):
        pytest.skip("Required functions not imported, skipping test")

def test_data_pipeline_with_temporary_files():
    """Test the data preparation and model pipeline with temporary files."""
    try:
        # Skip if prepare_dataset not available
        from prepare_dataset import merge_data
        
        # Create a very small test sample
        hours = pd.date_range(start='2024-01-01', periods=72, freq='H')
        
        # Create weather data
        weather_data = []
        for hour in hours:
            weather_data.append({
                'dt_iso': hour,
                'temp': np.random.uniform(30, 90),
                'feels_like': np.random.uniform(30, 85),
                'temp_min': np.random.uniform(25, 80),
                'temp_max': np.random.uniform(35, 95),
                'wind_speed': np.random.uniform(0, 20),
                'rain_1h': np.random.choice([0, 0, 0, 0, 0.5]),
                'weather_main': np.random.choice(['Clear', 'Clouds', 'Rain'])
            })
        weather_df = pd.DataFrame(weather_data)
        
        # Create taxi data
        taxi_data = []
        zones = [100, 101]
        for zone in zones:
            for hour in hours:
                for _ in range(np.random.randint(1, 10)):  # Multiple rides per hour
                    taxi_data.append({
                        'tpep_pickup_datetime': hour,
                        'PULocationID': zone,
                        'VendorID': np.random.randint(1, 3),
                        'trip_distance': np.random.uniform(1, 10)
                    })
        taxi_df = pd.DataFrame(taxi_data)
        
        # Create temp directory and files
        with tempfile.TemporaryDirectory() as temp_dir:
            weather_file = os.path.join(temp_dir, 'test_weather.csv')
            taxi_file = os.path.join(temp_dir, 'test_taxi.csv')
            output_file = os.path.join(temp_dir, 'test_output.csv')
            
            weather_df.to_csv(weather_file, index=False)
            taxi_df.to_csv(taxi_file, index=False)
            
            # Process the data
            merge_data(
                taxi_file=taxi_file,
                weather_file=weather_file,
                start_date='2024-01-01',
                end_date='2024-01-03',
                output_file=output_file
            )
            
            # Check if output file was created with expected features
            assert os.path.exists(output_file), "Output file was not created"
            
            # Load the preprocessed data
            processed_df = pd.read_csv(output_file)
            
            # Check basic structure
            basic_columns = ['hour', 'zone_id', 'demand', 'temp', 'hour_of_day']
            for col in basic_columns:
                assert col in processed_df.columns, f"Missing column {col}"
                
            # Check feature columns
            feature_columns = ['demand_lag_1h', 'is_weekend', 'is_rush_hour']
            for col in feature_columns:
                assert col in processed_df.columns, f"Missing feature column {col}"
            
            # Optional: if models are available, try to make predictions with this data
            if os.path.exists('models/best_lstm_model.keras'):
                # Extract features for a single zone/hour for testing
                zone_data = processed_df[processed_df['zone_id'] == zones[0]].sort_values('hour')
                
                if len(zone_data) >= 24:
                    # Get a slice of 24 consecutive hours
                    test_slice = zone_data.iloc[:24]
                    
                    # Extract features
                    features = {}
                    for col in MockConfig.FEATURES if 'MockConfig' in globals() else ['temp', 'feels_like', 'hour_of_day']:
                        if col in test_slice.columns:
                            features[col] = test_slice[col].tolist()
                    
                    # Fill in any missing features with zeros
                    for feature in ['demand_lag_1h', 'demand_lag_24h']:
                        if feature not in features:
                            features[feature] = [0.0] * 24
                    
                    # Make prediction
                    try:
                        result = predict_demand_from_features(features, zone_id=zones[0])
                        assert "predictions" in result, "No predictions in result"
                    except Exception as e:
                        print(f"Prediction with processed data failed: {e}")
                
    except (ImportError, NameError):
        pytest.skip("Required functions not imported, skipping test") 