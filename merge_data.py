import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os

def load_and_preprocess_weather(weather_file: str) -> pd.DataFrame:
    """
    Load and preprocess weather data.
    """
    print("Loading weather data...")
    weather_df = pd.read_csv(weather_file)
    
    # Convert timestamp to datetime
    weather_df['dt_iso'] = pd.to_datetime(weather_df['dt_iso'])
    
    # Round to nearest hour to match with taxi data
    weather_df['hour'] = weather_df['dt_iso'].dt.floor('h')
    
    # Select relevant features
    weather_features = [
        'hour', 'temp', 'feels_like', 'temp_min', 'temp_max',
        'wind_speed', 'rain_1h', 'weather_main'
    ]
    
    # Create dummy variables for weather_main
    weather_df = pd.get_dummies(weather_df, columns=['weather_main'], prefix='weather')
    
    return weather_df

def load_and_preprocess_taxi(taxi_file: str) -> pd.DataFrame:
    """
    Load and preprocess taxi data.
    """
    print("Loading taxi data...")
    taxi_df = pd.read_csv(taxi_file)
    
    # Convert timestamp to datetime
    taxi_df['tpep_pickup_datetime'] = pd.to_datetime(taxi_df['tpep_pickup_datetime'])
    
    # Round to nearest hour
    taxi_df['hour'] = taxi_df['tpep_pickup_datetime'].dt.floor('H')
    
    # Calculate demand metrics per hour
    demand_df = taxi_df.groupby('hour').agg({
        'VendorID': ['count', 'nunique'],  # Count trips and unique vendors per hour
        'trip_distance': ['mean', 'sum'],  # Average and total distance
        'PULocationID': lambda x: len(x.unique()),  # Number of unique pickup locations
        'is_holiday': 'first'  # Keep holiday information
    }).reset_index()
    
    # Flatten column names
    demand_df.columns = ['hour', 'trip_count', 'unique_vendors', 'avg_distance', 'total_distance', 'unique_pickup_locs', 'is_holiday']
    
    # Calculate normalized demand (0-100 scale)
    # Using a rolling window to get dynamic min/max values
    window_size = 168  # 1 week
    demand_df['rolling_max'] = demand_df['trip_count'].rolling(window=window_size, min_periods=1).max()
    demand_df['rolling_min'] = demand_df['trip_count'].rolling(window=window_size, min_periods=1).min()
    
    # Normalize demand to 0-100 scale
    demand_df['demand'] = 100 * (demand_df['trip_count'] - demand_df['rolling_min']) / (demand_df['rolling_max'] - demand_df['rolling_min'])
    
    # Handle edge cases
    demand_df['demand'] = demand_df['demand'].fillna(0)  # Fill NaN values
    demand_df['demand'] = demand_df['demand'].clip(0, 100)  # Clip values to 0-100 range
    
    # Drop intermediate columns
    demand_df = demand_df.drop(['rolling_max', 'rolling_min'], axis=1)
    
    # Validate demand values
    if demand_df['demand'].max() > 100 or demand_df['demand'].min() < 0:
        raise ValueError("Demand values outside expected range of 0-100")
    
    print(f"Demand statistics:")
    print(f"Average demand: {demand_df['demand'].mean():.2f}")
    print(f"Max demand: {demand_df['demand'].max():.2f}")
    print(f"Min demand: {demand_df['demand'].min():.2f}")
    
    return demand_df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to the dataframe.
    """
    # Extract time components
    df['hour_of_day'] = df['hour'].dt.hour
    df['day_of_week'] = df['hour'].dt.dayofweek
    df['month'] = df['hour'].dt.month
    df['day_of_month'] = df['hour'].dt.day
    df['week_of_year'] = df['hour'].dt.isocalendar().week
    
    # Create time period features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = df['hour_of_day'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    
    # Time periods
    df['time_period'] = pd.cut(
        df['hour_of_day'],
        bins=[-1, 6, 12, 18, 23],
        labels=['night', 'morning', 'afternoon', 'evening']
    )
    
    # Create dummy variables for time period
    df = pd.get_dummies(df, columns=['time_period'], prefix='period')
    
    return df

def merge_data(taxi_file: str, weather_file: str, output_file: str) -> None:
    """
    Merge taxi and weather data, add features, and save to file.
    """
    # Load and preprocess data
    weather_df = load_and_preprocess_weather(weather_file)
    demand_df = load_and_preprocess_taxi(taxi_file)
    
    print("Merging data...")
    # Merge on hour
    merged_df = pd.merge(demand_df, weather_df, on='hour', how='left')
    
    # Add time-based features
    print("Adding time features...")
    merged_df = add_time_features(merged_df)
    
    # Add lagged features
    print("Adding lagged features...")
    for lag in [1, 3, 24, 168]:  # 1 hour, 3 hours, 1 day, 1 week
        merged_df[f'demand_lag_{lag}h'] = merged_df['demand'].shift(lag)
        merged_df[f'avg_distance_lag_{lag}h'] = merged_df['avg_distance'].shift(lag)
    
    # Add rolling means
    print("Adding rolling means...")
    for window in [3, 6, 12, 24]:
        merged_df[f'demand_rolling_mean_{window}h'] = merged_df['demand'].rolling(window=window).mean()
        merged_df[f'avg_distance_rolling_mean_{window}h'] = merged_df['avg_distance'].rolling(window=window).mean()
    
    # Fill NaN values
    merged_df = merged_df.fillna(method='bfill').fillna(method='ffill')
    
    # Save to file
    print(f"Saving merged data to {output_file}...")
    merged_df.to_csv(output_file, index=False)
    
    print("\nData Summary:")
    print(f"Total hours: {len(merged_df)}")
    print(f"Date range: {merged_df['hour'].min()} to {merged_df['hour'].max()}")
    print(f"Total features: {len(merged_df.columns)}")
    print("\nFeature list:")
    for col in sorted(merged_df.columns):
        print(f"- {col}")

if __name__ == "__main__":
    # File paths
    taxi_file = "data_from_2024/cleaned_taxi_data.csv"
    weather_file = "data_from_2024/weather_data.csv"
    output_file = "data_from_2024/merged_features.csv"
    
    # Merge data
    merge_data(taxi_file, weather_file, output_file) 