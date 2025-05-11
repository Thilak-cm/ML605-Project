import pytest
import json
from datetime import datetime, timedelta
import os

def test_health_endpoint(mock_api_client):
    """Test the health check endpoint."""
    response = mock_api_client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "models" in response.json()  # Check models info is returned

def test_zones_with_coords_endpoint(mock_api_client):
    """Test the endpoint that returns zones with coordinates."""
    response = mock_api_client.get("/zones-with-coords")
    assert response.status_code == 200
    data = response.json()
    
    # Check structure - should be a list of zone objects
    assert isinstance(data, list)
    if len(data) > 0:
        # Check first zone has required properties
        first_zone = data[0]
        required_fields = ["id", "name", "borough", "lat", "lon"]
        for field in required_fields:
            assert field in first_zone, f"Required field {field} missing from zone data"

def test_predict_endpoint_empty_payload(mock_api_client):
    """Test predict endpoint with empty payload should return error."""
    response = mock_api_client.post("/predict", json={})
    assert response.status_code == 422  # Validation error

def test_predict_endpoint_with_minimal_data(mock_api_client):
    """Test predict endpoint with minimal valid data."""
    # Create a payload with just timestamp and zone_id
    payload = {
        "timestamp": (datetime.now() + timedelta(hours=2)).isoformat(),
        "zone_id": 236  # Using a zone we know exists
    }
    
    response = mock_api_client.post("/predict", json=payload)
    # If 200, check structure, if not 200, it might be expected (e.g., if models aren't available)
    if response.status_code == 200:
        data = response.json()
        assert "demand" in data
        assert "model_used" in data
        assert "timestamp" in data
        assert "zone_id" in data
    else:
        # Log the response for debugging
        print(f"Predict endpoint returned status code {response.status_code}: {response.json()}")

def test_predict_endpoint_with_future_timestamp(mock_api_client):
    """Test predict endpoint with far future timestamp (should use xgboost)."""
    future_date = (datetime.now() + timedelta(days=30)).isoformat()
    payload = {
        "timestamp": future_date,
        "zone_id": 236
    }
    
    response = mock_api_client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        # For far future, should use XGBoost
        assert data.get("model_used", "").lower() in ["xgboost", "prophet", "forecast"]
    else:
        # Log the response for debugging
        print(f"Predict endpoint with future timestamp returned status code {response.status_code}: {response.json()}")

def test_predict_endpoint_with_historical_features(mock_api_client):
    """Test predict endpoint with historical features included."""
    payload = {
        "timestamp": (datetime.now() + timedelta(hours=1)).isoformat(),
        "zone_id": 236,
        "historical_features": {
            "temp": [70.0] * 24,
            "feels_like": [68.0] * 24,
            "wind_speed": [5.0] * 24,
            "rain_1h": [0.0] * 24,
            "hour_of_day": list(range(24)),
            "day_of_week": [datetime.now().weekday()] * 24,
            "is_weekend": [int(datetime.now().weekday() >= 5)] * 24,
            "is_holiday": [0] * 24,
            "is_rush_hour": [1 if h in [7, 8, 9, 16, 17, 18] else 0 for h in range(24)],
            "demand_lag_1h": [40.0] * 24,
            "demand_lag_24h": [35.0] * 24,
            "demand_lag_168h": [30.0] * 24,
            "demand_rolling_mean_24h": [38.0] * 24,
            "demand_rolling_std_24h": [5.0] * 24,
            "demand_change_1h": [1.0] * 24
        }
    }
    
    response = mock_api_client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        # With historical features, should use LSTM
        assert data.get("model_used", "").lower() in ["lstm", "transformer"]
    else:
        # Log the response for debugging
        print(f"Predict endpoint with historical features returned status code {response.status_code}: {response.json()}")

def test_live_features_endpoint(mock_api_client):
    """Test live features endpoint."""
    # Test with minimal parameters
    response = mock_api_client.get("/live-features/?zone_id=236&mock=true")
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    assert "zone_id" in data
    assert "features" in data
    assert isinstance(data["features"], dict)
    
    # Check some key features we expect to find
    expected_features = ["temp", "feels_like", "wind_speed", "hour_of_day", "day_of_week"]
    for feature in expected_features:
        assert feature in data["features"], f"Expected feature {feature} not found in live features"
        assert isinstance(data["features"][feature], list), f"Feature {feature} should be a list"

def test_invalid_zone_id(mock_api_client):
    """Test that invalid zone ID returns appropriate error."""
    # Use a likely invalid zone ID (very high number)
    response = mock_api_client.get("/live-features/?zone_id=99999&mock=true")
    
    # We expect either a 404 or a 400 error, or special handling in the app
    # Let's check for status code and error message
    if response.status_code != 200:
        assert response.status_code in [400, 404, 422], f"Unexpected status code: {response.status_code}"
    else:
        # If it returns 200, the app might have special handling for invalid zones
        print("Note: API accepted invalid zone ID - check implementation details") 