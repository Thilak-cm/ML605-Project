import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import holidays
import os

def load_and_preprocess_weather(weather_file: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load and preprocess weather data.
    """
    print("Loading weather data...")
    weather_df = pd.read_csv(weather_file)
    
    # Convert timestamp to datetime
    weather_df['dt_iso'] = pd.to_datetime(weather_df['dt_iso'])

    # filter for the certain dates
    weather_df = weather_df[(weather_df['dt_iso'] >= start_date) & (weather_df['dt_iso'] <= end_date)]
    
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

def load_and_preprocess_taxi(taxi_file: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load and preprocess taxi data with zone-level demand aggregation.
    """
    print("Loading taxi data...")
    taxi_df = pd.read_csv(taxi_file)
    
    # Convert timestamp to datetime
    taxi_df['tpep_pickup_datetime'] = pd.to_datetime(taxi_df['tpep_pickup_datetime'])

    # filter for the certain dates
    taxi_df = taxi_df[(taxi_df['tpep_pickup_datetime'] >= start_date) & (taxi_df['tpep_pickup_datetime'] <= end_date)]
    
    # Round to nearest hour
    taxi_df['hour'] = taxi_df['tpep_pickup_datetime'].dt.floor('h')
    
    # Rename PULocationID to zone_id
    taxi_df = taxi_df.rename(columns={'PULocationID': 'zone_id'})
    
    # Get US holidays
    us_holidays = holidays.US()
    taxi_df['is_holiday'] = taxi_df['tpep_pickup_datetime'].dt.date.map(lambda x: x in us_holidays).astype(int)
    
    # Calculate demand metrics per hour and zone
    print("Calculating zone-level demand metrics...")
    demand_df = taxi_df.groupby(['hour', 'zone_id']).agg({
        'VendorID': ['count', 'nunique'],  # Count trips and unique vendors
        'trip_distance': ['mean', 'sum'],  # Average and total distance
        'is_holiday': 'first'  # Keep holiday information
    }).reset_index()
    
    # Flatten column names
    demand_df.columns = ['hour', 'zone_id', 'trip_count', 'unique_vendors', 
                        'avg_distance', 'total_distance', 'is_holiday']
    
    # Keep raw trip count as primary demand metric
    demand_df['demand'] = demand_df['trip_count']
    
    # Sort by zone and hour for correct lag calculation
    demand_df = demand_df.sort_values(['zone_id', 'hour'])
    
    # Add demand normalization features using only historical data
    print("Adding demand normalization features...")
    grouped = demand_df.groupby('zone_id')
    
    # 1. Log transform (safe to use as it's a point-wise operation)
    demand_df['demand_log'] = np.log1p(demand_df['demand'])
    
    # 2. Rolling z-score normalization (using only historical data)
    window_size = 168  # 1 week
    demand_df['demand_rolling_mean'] = grouped['demand'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    demand_df['demand_rolling_std'] = grouped['demand'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).std()
    )
    demand_df['demand_zscore'] = (demand_df['demand'] - demand_df['demand_rolling_mean']) / \
                                (demand_df['demand_rolling_std'] + 1e-8)
    
    # 3. Rolling min-max scaling (using only historical data)
    demand_df['demand_rolling_min'] = grouped['demand'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).min()
    )
    demand_df['demand_rolling_max'] = grouped['demand'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).max()
    )
    demand_df['demand_minmax'] = (demand_df['demand'] - demand_df['demand_rolling_min']) / \
                                (demand_df['demand_rolling_max'] - demand_df['demand_rolling_min'] + 1e-8)
    
    print("\nDemand statistics by zone:")
    stats_cols = ['demand', 'demand_log', 'demand_zscore', 'demand_minmax']
    for col in stats_cols:
        print(f"\n{col} statistics:")
        zone_stats = demand_df.groupby('zone_id')[col].agg(['mean', 'std', 'min', 'max']).round(2)
        print(zone_stats.describe())
    
    return demand_df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to the dataframe.
    """
    print("Adding time features...")
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

def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged and rolling features per zone.
    """
    print("Adding zone-level lagged features...")
    # Sort by zone and hour for correct lag calculation
    df = df.sort_values(['zone_id', 'hour'])
    
    # Group by zone for zone-specific calculations
    grouped = df.groupby('zone_id')
    
    # Add lagged features for both raw and normalized demand
    for lag in [1, 3, 24, 168]:  # 1 hour, 3 hours, 1 day, 1 week
        # Raw demand lags
        df[f'demand_lag_{lag}h'] = grouped['demand'].transform(lambda x: x.shift(lag))
        # Log demand lags
        df[f'demand_log_lag_{lag}h'] = grouped['demand_log'].transform(lambda x: x.shift(lag))
        # Distance lags
        df[f'avg_distance_lag_{lag}h'] = grouped['avg_distance'].transform(lambda x: x.shift(lag))
    
    # Add rolling means
    for window in [3, 6, 12, 24]:
        # Raw demand rolling stats
        df[f'demand_rolling_mean_{window}h'] = grouped['demand'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'demand_rolling_std_{window}h'] = grouped['demand'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        # Log demand rolling mean
        df[f'demand_log_rolling_mean_{window}h'] = grouped['demand_log'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        # Distance rolling mean
        df[f'avg_distance_rolling_mean_{window}h'] = grouped['avg_distance'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    # Add demand changes (both raw and log)
    df['demand_change_1h'] = df['demand'] - df['demand_lag_1h']
    df['demand_log_change_1h'] = df['demand_log'] - df['demand_log_lag_1h']
    
    return df

def merge_data(taxi_file: str, weather_file: str, start_date: str, end_date: str, output_file: str) -> None:
    """
    Merge taxi and weather data, add features, and save to file.
    """
    # Load and preprocess data
    weather_df = load_and_preprocess_weather(weather_file, start_date, end_date)
    demand_df = load_and_preprocess_taxi(taxi_file, start_date, end_date)
    
    print("Merging weather data...")
    # Merge on hour (weather data will be duplicated for each zone)
    merged_df = pd.merge(demand_df, weather_df, on='hour', how='left')
    
    # Add time-based features
    merged_df = add_time_features(merged_df)
    
    # Add lagged and rolling features
    merged_df = add_lagged_features(merged_df)
    
    # Drop redundant columns
    redundant_cols = [
        'trip_count',  # Same as demand
        'dt_iso',      # Already captured in temporal features
        'weather_description',  # Already have weather types
        'weather_id'   # Redundant with weather types
    ]
    merged_df = merged_df.drop(columns=redundant_cols, errors='ignore')
    
    # Fill NaN values using forward fill then backward fill
    merged_df = merged_df.ffill().bfill()
    
    # Save to file
    print(f"\nSaving merged data to {output_file}...")
    merged_df.to_csv(output_file, index=False)
    
    print("\nData Summary:")
    print(f"Total records: {len(merged_df)}")
    print(f"Unique zones: {merged_df['zone_id'].nunique()}")
    print(f"Date range: {merged_df['hour'].min()} to {merged_df['hour'].max()}")
    
    # Print final feature list
    print("\nFinal model input features:")
    # Exclude 'hour' as it's not a model input
    feature_cols = [col for col in merged_df.columns if col != 'hour']
    
    # Group features by type for better readability
    feature_groups = {
        'Demand Features': [col for col in feature_cols if 'demand' in col],
        'Weather Features': [col for col in feature_cols if any(x in col for x in ['temp', 'rain', 'wind', 'weather_'])],
        'Time Features': [col for col in feature_cols if any(x in col for x in ['hour', 'day', 'week', 'month', 'period'])],
        'Trip Features': [col for col in feature_cols if any(x in col for x in ['distance', 'vendors'])],
        'Other Features': [col for col in feature_cols if not any(x in col for x in ['demand', 'temp', 'rain', 'wind', 'weather_', 'hour', 'day', 'week', 'month', 'period', 'distance', 'vendors'])]
    }
    
    for group, features in feature_groups.items():
        if features:
            print(f"\n{group}:")
            for feature in sorted(features):
                print(f"- {feature}")

if __name__ == "__main__":
    # File paths
    taxi_file = "data_from_2024/cleaned_taxi_data_short.csv"
    weather_file = "data_from_2024/weather_data.csv"
    output_file = "data_from_2024/taxi_demand_dataset.csv"
    
    # Merge data with a specific time window, just take 2024 to 2025 as an example
    start_date = "2024-01-01"
    end_date = "2025-01-01"
    merge_data(taxi_file, weather_file, start_date, end_date, output_file) 