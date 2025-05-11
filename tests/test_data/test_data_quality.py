import pytest
import pandas as pd
import numpy as np
import os

def test_dataset_file_exists():
    """Test that the main dataset file exists."""
    data_path = os.path.join('data_from_2024', 'taxi_demand_dataset.csv')
    assert os.path.exists(data_path), f"Dataset file not found at {data_path}"

def test_zone_lookup_file_exists():
    """Test that the zone lookup file exists."""
    assert os.path.exists('zone_lookup_lat_long.csv'), "Zone lookup file not found"
    assert os.path.exists('taxi_zone_lookup.csv'), "Taxi zone lookup file not found"

def test_dataset_has_required_columns(test_data_path):
    """Test that dataset has all required columns."""
    try:
        data_path = os.path.join(test_data_path, 'taxi_demand_dataset.csv')
        df = pd.read_csv(data_path)
        
        required_columns = ['hour', 'zone_id', 'demand']
        for col in required_columns:
            assert col in df.columns, f"Required column {col} missing from dataset"
    except FileNotFoundError:
        pytest.skip("Dataset file not found, skipping test")

def test_zone_lookup_has_required_columns():
    """Test that zone lookup file has all required columns."""
    try:
        df = pd.read_csv('zone_lookup_lat_long.csv')
        
        required_columns = ['LocationID', 'Latitude', 'Longitude']
        for col in required_columns:
            assert col in df.columns, f"Required column {col} missing from zone lookup"
    except FileNotFoundError:
        pytest.skip("Zone lookup file not found, skipping test")

def test_dataset_zone_ids_exist_in_lookup():
    """Test that all zone IDs in the dataset exist in the zone lookup."""
    try:
        dataset = pd.read_csv(os.path.join('data_from_2024', 'taxi_demand_dataset.csv'))
        zone_lookup = pd.read_csv('zone_lookup_lat_long.csv')
        
        dataset_zones = set(dataset['zone_id'].unique())
        lookup_zones = set(zone_lookup['LocationID'].unique())
        
        excluded_zones = {264, 265} # because we don't have the coordinates for these zones
        missing_zones = dataset_zones - lookup_zones - excluded_zones
        assert len(missing_zones) == 0, f"Dataset contains zone IDs not in zone lookup: {missing_zones}"
    except FileNotFoundError:
        pytest.skip("One or both files not found, skipping test")

def test_data_types(test_data_path):
    """Test that dataset columns have the expected data types."""
    try:
        data_path = os.path.join(test_data_path, 'taxi_demand_dataset.csv')
        df = pd.read_csv(data_path)
        
        # Check that zone_id is an integer
        assert pd.api.types.is_integer_dtype(df['zone_id']), "zone_id should be an integer type"
        
        # Check that demand is numeric
        assert pd.api.types.is_numeric_dtype(df['demand']), "demand should be a numeric type"
        
        # Check that hour can be converted to datetime
        try:
            pd.to_datetime(df['hour'])
        except:
            assert False, "hour column cannot be converted to datetime"
    except FileNotFoundError:
        pytest.skip("Dataset file not found, skipping test")

def test_demand_values_are_non_negative(test_data_path):
    """Test that demand values are non-negative."""
    try:
        data_path = os.path.join(test_data_path, 'taxi_demand_dataset.csv')
        df = pd.read_csv(data_path)
        
        assert (df['demand'] >= 0).all(), "Some demand values are negative"
    except FileNotFoundError:
        pytest.skip("Dataset file not found, skipping test") 