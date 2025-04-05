import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Dict, Optional, List, Tuple
from clearml import Task, Dataset, OutputModel
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Global variable to store the loaded model
_model: Optional[Prophet] = None

def create_evaluation_plots(y_true: np.ndarray, y_pred: np.ndarray, 
                          forecast_df: pd.DataFrame,
                          task: Task) -> None:
    """
    Create and log evaluation plots to ClearML
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        forecast_df: Prophet forecast DataFrame
        task: ClearML task object
    """
    logger = task.get_logger()
    
    # Create residual plot
    plt.figure(figsize=(12, 6))
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Demand')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    logger.report_matplotlib_figure(
        title="Residual Plot",
        series="Evaluation",
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    # Create actual vs predicted plot
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Demand')
    plt.ylabel('Predicted Demand')
    plt.title('Actual vs Predicted Demand')
    logger.report_matplotlib_figure(
        title="Actual vs Predicted",
        series="Evaluation",
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    # Create time series plot
    plt.figure(figsize=(15, 6))
    plt.plot(forecast_df['ds'], y_true, label='Actual', alpha=0.5)
    plt.plot(forecast_df['ds'], y_pred, label='Predicted', alpha=0.5)
    plt.fill_between(forecast_df['ds'], 
                    forecast_df['yhat_lower'], 
                    forecast_df['yhat_upper'], 
                    alpha=0.2, label='95% Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.title('Time Series Forecast')
    plt.legend()
    logger.report_matplotlib_figure(
        title="Time Series Forecast",
        series="Evaluation",
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    # Create components plot if available
    try:
        from prophet.plot import plot_components
        plt.figure(figsize=(15, 10))
        plot_components(forecast_df)
        logger.report_matplotlib_figure(
            title="Forecast Components",
            series="Evaluation",
            figure=plt.gcf(),
            iteration=0
        )
        plt.close()
    except Exception as e:
        print(f"Could not create components plot: {str(e)}")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the merged features data for Prophet model. Prophet requires specific column names:
    'ds' for datetime and 'y' for the target variable.
    
    Args:
        df: Merged features DataFrame with normalized demand values
        
    Returns:
        Processed DataFrame ready for Prophet model
    """
    # Rename hour column to ds for Prophet
    prophet_df = df.copy()
    prophet_df = prophet_df.rename(columns={'hour': 'ds'})
    
    # Add time-based features that Prophet can use as regressors
    prophet_df['hour_of_day'] = prophet_df['ds'].dt.hour
    prophet_df['day_of_week'] = prophet_df['ds'].dt.dayofweek
    prophet_df['is_weekend'] = prophet_df['day_of_week'].isin([5, 6]).astype(int)
    prophet_df['is_rush_hour'] = prophet_df['hour_of_day'].isin([7,8,9,16,17,18,19]).astype(int)
    
    # Add weather-based features as regressors
    weather_features = ['temp', 'feels_like', 'wind_speed', 'rain_1h', 'weather_id']
    
    # Add distance and demand-based features
    demand_features = ['demand_lag_1h', 'demand_lag_2h', 'demand_lag_3h', 
                      'demand_lag_24h', 'demand_lag_168h']
    distance_features = ['avg_distance_lag_1h', 'avg_distance_lag_2h', 'avg_distance_lag_3h',
                        'avg_distance_lag_24h', 'avg_distance_lag_168h']
    
    # Select final features for Prophet
    selected_features = ['ds', 'demand'] + weather_features + demand_features + distance_features + \
                       ['is_weekend', 'is_rush_hour', 'day_of_week', 'hour_of_day']
    
    # Keep only features that exist in the dataframe
    selected_features = [f for f in selected_features if f in prophet_df.columns]
    prophet_df = prophet_df[selected_features]
    
    # Print feature information for debugging
    print(f"Total features used: {len(selected_features)}")
    print("Features:", selected_features)
    
    # Rename demand to y for Prophet
    prophet_df = prophet_df.rename(columns={'demand': 'y'})
    
    # Sort by timestamp
    prophet_df = prophet_df.sort_values('ds')
    
    return prophet_df

def evaluate_model(model: Prophet, test_data: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate the model on test data and return metrics.
    
    Args:
        model: Trained Prophet model
        test_data: Test dataset
        
    Returns:
        Tuple of (metrics dictionary, forecast DataFrame)
    """
    # Make predictions on test data
    forecast = model.predict(test_data)
    
    # Calculate metrics on original scale
    mse = mean_squared_error(test_data['y'], forecast['yhat'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data['y'], forecast['yhat'])
    r2 = r2_score(test_data['y'], forecast['yhat'])
    mape = np.mean(np.abs((test_data['y'] - forecast['yhat']) / test_data['y'])) * 100
    
    # Calculate metrics for different time periods
    hourly_metrics = {}
    for hour in range(24):
        # Get indices for the current hour
        hour_mask = pd.to_datetime(test_data['ds']).dt.hour == hour
        if hour_mask.any():
            y_true_hour = test_data.loc[hour_mask, 'y'].values
            y_pred_hour = forecast.loc[forecast.index[hour_mask], 'yhat'].values
            hour_mse = mean_squared_error(y_true_hour, y_pred_hour)
            hourly_metrics[f'hour_{hour}_rmse'] = np.sqrt(hour_mse)
    
    # Combine all metrics
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        **hourly_metrics
    }
    
    return metrics, forecast

def train_and_save_model(data_path: str = 'data_from_2024/merged_features.csv', 
                        model_path: str = 'prophet_model.joblib') -> None:
    """
    Train a Prophet model on the provided data and save it to disk.
    Also logs experiment to ClearML.
    
    Args:
        data_path: Path to the merged features CSV file with normalized demand values
        model_path: Path where the model should be saved
    """
    # Initialize ClearML task
    task = Task.init(
        project_name='TaxiDemandPrediction',
        task_name='ProphetTraining_Normalized',
        task_type='training'
    )
    logger = task.get_logger()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    df['hour'] = pd.to_datetime(df['hour'])
    
    # Validate demand values are normalized
    demand_min = df['demand'].min()
    demand_max = df['demand'].max()
    if demand_min < 0 or demand_max > 100:
        print(f"Warning: Demand values outside expected normalized range [0,100]. Min: {demand_min}, Max: {demand_max}")
    
    # Preprocess data
    print("Preprocessing data...")
    prophet_df = preprocess_data(df)
    
    # Split data into train and test sets (last 2 weeks for testing)
    test_size = 14 * 24  # 14 days * 24 hours
    train_df = prophet_df[:-test_size]
    test_df = prophet_df[-test_size:]
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Initialize and train Prophet model with additional regressors
    print("Training model...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative',
        interval_width=0.95
    )
    
    # Add all features except 'ds' and 'y' as regressors
    regressor_columns = [col for col in train_df.columns if col not in ['ds', 'y']]
    for column in regressor_columns:
        model.add_regressor(column)
    
    # Fit the model
    model.fit(train_df)
    
    # Evaluate model
    print("Evaluating model...")
    metrics, forecast_df = evaluate_model(model, test_df)
    
    # Log metrics
    for metric_name, metric_value in metrics.items():
        logger.report_scalar(
            title='Evaluation Metrics',
            series=metric_name,
            value=metric_value,
            iteration=0
        )
    
    # Create evaluation plots
    create_evaluation_plots(
        test_df['y'].values,
        forecast_df['yhat'].values,
        forecast_df,
        task
    )
    
    # Save model and feature list
    print(f"Saving model to {model_path}...")
    model_data = {
        'model': model,
        'features': regressor_columns,
        'metrics': metrics
    }
    joblib.dump(model_data, model_path)
    
    # Upload model to ClearML
    output_model = OutputModel(
        task=task,
        name='prophet_model'
    )
    output_model.update_weights(model_path)
    
    print("Training complete!")
    task.close()

def load_model(model_path: str = 'prophet_model.joblib') -> Dict:
    """
    Load the trained model and components from disk.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Dictionary containing model and its components
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    return joblib.load(model_path)

def predict_demand(input_features: Dict[str, any], forecast_hours: int = 24) -> Dict[str, List[float]]:
    """
    Predict taxi demand for the next N hours using the trained Prophet model.
    
    Args:
        input_features: Dictionary containing feature values for prediction
                       All values should be normalized/scaled appropriately
        forecast_hours: Number of hours to forecast (default: 24)
        
    Returns:
        Dictionary containing predicted demand values and confidence intervals
    """
    global _model
    
    # Load model if not already loaded
    if _model is None:
        model_data = load_model()
        _model = model_data['model']
        required_features = model_data['features']
        print("Required features:", required_features)
    
    # Validate input features
    missing_features = [f for f in required_features if f not in input_features]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Create future dataframe for prediction
    future_dates = pd.date_range(
        start=input_features['ds'],
        periods=forecast_hours,
        freq='H'
    )
    
    # Create prediction dataframe with all required features
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Add time-based features
    future_df['hour_of_day'] = future_df['ds'].dt.hour
    future_df['day_of_week'] = future_df['ds'].dt.dayofweek
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
    future_df['is_rush_hour'] = future_df['hour_of_day'].isin([7,8,9,16,17,18,19]).astype(int)
    
    # Add other features from input_features
    for feature in required_features:
        if feature not in future_df.columns:
            future_df[feature] = input_features.get(feature, 0)
    
    # Make prediction
    forecast = _model.predict(future_df)
    
    # Ensure predictions are within normalized range [0, 100]
    forecast['yhat'] = np.clip(forecast['yhat'], 0, 100)
    forecast['yhat_lower'] = np.clip(forecast['yhat_lower'], 0, 100)
    forecast['yhat_upper'] = np.clip(forecast['yhat_upper'], 0, 100)
    
    # Return predictions
    return {
        'timestamps': future_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'demand': forecast['yhat'].tolist(),
        'demand_lower': forecast['yhat_lower'].tolist(),
        'demand_upper': forecast['yhat_upper'].tolist()
    }

if __name__ == "__main__":
    # Train model on the merged features dataset
    train_and_save_model()
    
    # Get current time for test prediction
    current_time = datetime.now()
    
    # Test prediction with all required features
    test_features = {
        'ds': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        # Weather features
        'temp': 20.0,
        'feels_like': 19.0,
        'wind_speed': 5.0,
        'rain_1h': 0.0,
        'weather_id': 800,  # Clear sky
        # Demand lag features (normalized values)
        'demand_lag_1h': 50.0,
        'demand_lag_3h': 45.0,
        'demand_lag_24h': 55.0,
        'demand_lag_168h': 52.0,  # Week ago
        # Distance lag features (normalized values)
        'avg_distance_lag_1h': 40.0,
        'avg_distance_lag_3h': 42.0,
        'avg_distance_lag_24h': 45.0,
        'avg_distance_lag_168h': 43.0,
        # Time-based features
        'hour_of_day': current_time.hour,
        'day_of_week': current_time.weekday(),
        'is_weekend': int(current_time.weekday() >= 5),
        'is_rush_hour': int(current_time.hour in [7,8,9,16,17,18,19])
    }
    
    # Make prediction
    print("\nMaking test prediction with features:")
    for feature, value in test_features.items():
        print(f"{feature}: {value}")
        
    prediction = predict_demand(test_features)
    
    print("\nPredictions for next 24 hours:")
    for i, (ts, pred, lower, upper) in enumerate(zip(
        prediction['timestamps'],
        prediction['demand'],
        prediction['demand_lower'],
        prediction['demand_upper']
    )):
        print(f"Hour {i+1}: {ts} - Demand: {pred:.2f} [{lower:.2f}, {upper:.2f}]") 