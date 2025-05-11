import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

@pytest.fixture(scope="session")
def test_data_path():
    """Fixture to provide the path to test data files."""
    return Path("data_from_2024")

@pytest.fixture(scope="session")
def sample_data():
    """Fixture to provide a small sample dataset for tests."""
    # Create a sample dataset with key columns found in the main dataset
    data = {
        'hour': pd.date_range(start='2024-01-01', periods=48, freq='H'),
        'zone_id': np.random.choice([100, 101, 102, 103], size=48),
        'demand': np.random.randint(10, 100, size=48),
        'temp': np.random.uniform(30, 90, size=48),
        'feels_like': np.random.uniform(30, 85, size=48),
        'wind_speed': np.random.uniform(0, 20, size=48),
        'rain_1h': np.random.choice([0, 0, 0, 0, 1], size=48),  # Mostly no rain
        'hour_of_day': [h.hour for h in pd.date_range(start='2024-01-01', periods=48, freq='H')],
        'day_of_week': [h.dayofweek for h in pd.date_range(start='2024-01-01', periods=48, freq='H')],
        'is_weekend': [int(h.dayofweek >= 5) for h in pd.date_range(start='2024-01-01', periods=48, freq='H')],
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_test_file():
    """Fixture to create a temporary CSV file with test data."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
        # Create a sample dataset
        data = {
            'hour': pd.date_range(start='2024-01-01', periods=24, freq='H'),
            'zone_id': np.random.choice([100, 101], size=24),
            'demand': np.random.randint(10, 100, size=24),
            'temp': np.random.uniform(30, 90, size=24),
        }
        df = pd.DataFrame(data)
        # Save to the temporary file
        df.to_csv(temp.name, index=False)
        temp_path = temp.name
    
    # Return the path to the temporary file
    yield temp_path
    
    # Clean up after the test
    if os.path.exists(temp_path):
        os.remove(temp_path)

@pytest.fixture(scope="session")
def mock_api_client():
    """Fixture to provide a mock API client for testing API endpoints."""
    try:
        from app import app
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI not installed or app.py not found")