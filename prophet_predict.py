import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Dict, Optional, List, Tuple
from clearml import Task, Dataset, OutputModel
import joblib
import os
from datetime import datetime, timedelta
import glob

# Global variable to store the loaded model
_model: Optional[Prophet] = None

def load_and_merge_data(data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge all parquet files, then split into train and test sets.
    Train set will be all data except the last month, which will be the test set.
    
    Args:
        data_dir: Directory containing the parquet files
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Get all parquet files
    parquet_files = sorted(glob.glob(os.path.join(data_dir, 'yellow_tripdata_*.parquet')))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load all files
    all_dfs = []
    for file in parquet_files:
        print(f"Loading {os.path.basename(file)}...")
        df = pd.read_parquet(file)
        all_dfs.append(df)
    
    # Combine all dataframes
    print("Merging dataframes...")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by pickup datetime
    merged_df = merged_df.sort_values('tpep_pickup_datetime')
    
    # Split into train and test
    # Use the last month (2025-01) as test set
    split_date = pd.Timestamp('2025-01-01')
    
    train_df = merged_df[merged_df['tpep_pickup_datetime'] < split_date]
    test_df = merged_df[merged_df['tpep_pickup_datetime'] >= split_date]
    
    print(f"Train set: {len(train_df):,} records from {train_df['tpep_pickup_datetime'].min()} to {train_df['tpep_pickup_datetime'].max()}")
    print(f"Test set: {len(test_df):,} records from {test_df['tpep_pickup_datetime'].min()} to {test_df['tpep_pickup_datetime'].max()}")
    
    return train_df, test_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the taxi data for Prophet model. Prophet requires specific column names:
    'ds' for datetime and 'y' for the target variable.
    
    Args:
        df: Raw taxi trip data DataFrame
        
    Returns:
        Processed DataFrame ready for Prophet model
    """
    # Convert pickup datetime to hourly frequency
    df['hour'] = df['tpep_pickup_datetime'].dt.floor('H')
    
    # Calculate demand (number of pickups per hour)
    demand_df = df.groupby('hour').size().reset_index()
    demand_df.columns = ['ds', 'y']  # Prophet requires these column names
    
    # Sort by timestamp
    demand_df = demand_df.sort_values('ds')
    
    return demand_df

def evaluate_model(model: Prophet, test_data: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate the model on test data and return metrics.
    
    Args:
        model: Trained Prophet model
        test_data: Test dataset
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions on test data
    forecast = model.predict(test_data)
    
    # Calculate metrics
    mse = np.mean((forecast['yhat'] - test_data['y'])**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test_data['y'] - forecast['yhat']) / test_data['y'])) * 100
    
    # Calculate R-squared
    y_mean = test_data['y'].mean()
    ss_tot = np.sum((test_data['y'] - y_mean)**2)
    ss_res = np.sum((test_data['y'] - forecast['yhat'])**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

def train_and_save_model(data_dir: str = 'data', model_path: str = 'prophet_model.joblib') -> None:
    """
    Train a Prophet model on the provided data and save it to disk.
    Also logs experiment to ClearML.
    
    Args:
        data_dir: Directory containing the parquet files
        model_path: Path where the model should be saved
    """
    # Initialize ClearML task
    task = Task.init(
        project_name='TaxiDemandPrediction',
        task_name='ProphetTraining_MultiMonth',
        task_type='training'
    )
    
    # Load and preprocess data
    print("Loading and merging data...")
    train_df, test_df = load_and_merge_data(data_dir)
    
    # Preprocess train and test data
    train_data = preprocess_data(train_df)
    test_data = preprocess_data(test_df)
    
    # Log dataset info
    task.connect({
        "dataset_info": {
            "train_size": len(train_data),
            "test_size": len(test_data),
            "train_date_range": {
                "start": train_data['ds'].min().strftime('%Y-%m-%d %H:%M'),
                "end": train_data['ds'].max().strftime('%Y-%m-%d %H:%M')
            },
            "test_date_range": {
                "start": test_data['ds'].min().strftime('%Y-%m-%d %H:%M'),
                "end": test_data['ds'].max().strftime('%Y-%m-%d %H:%M')
            }
        }
    })
    
    # Configure Prophet model with parameters suitable for hourly data
    model_params = {
        'yearly_seasonality': True,   # Now we have enough data for yearly patterns
        'weekly_seasonality': True,   # Strong weekly patterns in taxi demand
        'daily_seasonality': True,    # Strong daily patterns
        'seasonality_mode': 'multiplicative',  # Taxi demand typically has multiplicative seasonality
        'interval_width': 0.95,       # 95% prediction intervals
        'changepoint_prior_scale': 0.05,  # Flexibility of trend changes
    }
    
    # Log parameters
    task.connect({"model_params": model_params})
    
    # Initialize and train model
    print("Training model...")
    model = Prophet(**model_params)
    
    # Add additional regressors or seasonality if needed
    # model.add_seasonality(name='rush_hour', period=0.5, fourier_order=3)
    
    model.fit(train_data)
    
    # Evaluate on test data
    print("Evaluating model...")
    metrics = evaluate_model(model, test_data)
    
    # Log metrics
    logger = task.get_logger()
    for metric_name, value in metrics.items():
        logger.report_scalar("Metrics", metric_name, iteration=0, value=value)
        print(f"{metric_name.upper()}: {value:.2f}")
    
    # Save model components
    model_components = {
        'model': model,
        'metrics': metrics,
        'params': model_params
    }
    
    # Save model locally
    joblib.dump(model_components, model_path)
    print(f"Model saved locally to {model_path}")
    
    # Register model in ClearML
    output_model = OutputModel(
        task=task,
        name='TaxiDemandProphet_MultiMonth',
        framework='prophet',
        label_enumeration={}
    )
    output_model.update_weights(weights_filename=model_path)
    
    print("Model registered in ClearML")
    
    # Close the task
    task.close()

def load_model(model_path: str = 'prophet_model.joblib') -> Prophet:
    """
    Load the trained Prophet model from disk. Uses a global variable to cache the model.
    If model doesn't exist locally, attempts to download from ClearML.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Loaded Prophet model
    """
    global _model
    
    if _model is None:
        if not os.path.exists(model_path):
            print("Model not found locally, attempting to download from ClearML...")
            try:
                task = Task.get_task(project_name='TaxiDemandPrediction')
                model_id = task.models['output'][-1].id
                output_model = OutputModel(model_id=model_id)
                model_path = output_model.get_weights()
                print(f"Model downloaded from ClearML to {model_path}")
            except Exception as e:
                raise FileNotFoundError(f"Could not find model locally or in ClearML: {str(e)}")
        
        model_components = joblib.load(model_path)
        _model = model_components['model']
    
    return _model

def predict_demand(input_features: Dict[str, any], forecast_hours: int = 24) -> Dict[str, List[float]]:
    """
    Predict taxi demand for the next n hours based on input features.
    
    Args:
        input_features: Dictionary containing the current timestamp
        forecast_hours: Number of hours to forecast into the future
        
    Returns:
        Dictionary containing forecasted demand and confidence intervals
    """
    # Ensure timestamp is provided
    if 'timestamp' not in input_features:
        raise ValueError("Missing required feature: timestamp")
    
    # Convert timestamp to datetime if string
    current_time = pd.to_datetime(input_features['timestamp'])
    
    # Create future dataframe for prediction
    future_dates = pd.date_range(
        start=current_time,
        periods=forecast_hours,
        freq='H'
    )
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Load model and make prediction
    model = load_model()
    forecast = model.predict(future_df)
    
    # Return predictions with confidence intervals
    return {
        'timestamp': forecast['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'demand': forecast['yhat'].tolist(),
        'demand_lower': forecast['yhat_lower'].tolist(),
        'demand_upper': forecast['yhat_upper'].tolist()
    }

if __name__ == "__main__":
    # Train model on all available data
    train_and_save_model()
    
    # Test prediction
    test_features = {
        'timestamp': '2025-02-01 00:00:00'
    }
    
    prediction = predict_demand(test_features)
    print("\nTest prediction for next 24 hours:")
    for i in range(len(prediction['timestamp'])):
        print(f"Time: {prediction['timestamp'][i]}")
        print(f"Demand: {prediction['demand'][i]:.2f} [{prediction['demand_lower'][i]:.2f}, {prediction['demand_upper'][i]:.2f}]") 