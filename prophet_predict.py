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
        df: Merged features DataFrame
        
    Returns:
        Processed DataFrame ready for Prophet model
    """
    # Rename hour column to ds for Prophet
    prophet_df = df.copy()
    prophet_df = prophet_df.rename(columns={'hour': 'ds'})
    
    # Add time-based features that Prophet can use as regressors
    prophet_df['hour_of_day'] = prophet_df['ds'].dt.hour
    prophet_df['is_weekend'] = prophet_df['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    prophet_df['is_rush_hour'] = prophet_df['hour_of_day'].isin([7,8,9,16,17,18,19]).astype(int)
    
    # Add weather-based features as regressors
    weather_features = ['temp', 'feels_like', 'wind_speed', 'rain_1h']
    
    # Add distance-based features
    distance_features = ['avg_distance', 'total_distance', 'unique_pickup_locs']
    
    # Select final features for Prophet
    selected_features = ['ds', 'demand'] + weather_features + distance_features + ['is_weekend', 'is_rush_hour']
    prophet_df = prophet_df[selected_features]
    
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
        data_path: Path to the merged features CSV file
        model_path: Path where the model should be saved
    """
    # Initialize ClearML task
    task = Task.init(
        project_name='TaxiDemandPrediction',
        task_name='ProphetTraining_Improved',
        task_type='training'
    )
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    df['hour'] = pd.to_datetime(df['hour'])
    
    # Preprocess data
    print("Preprocessing data...")
    data = preprocess_data(df)
    
    # Split into train and test (last 20% for testing)
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
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
    
    # Configure Prophet model
    model_params = {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'seasonality_mode': 'multiplicative',
        'interval_width': 0.95,
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10,
        'holidays_prior_scale': 10,
    }
    
    # Log parameters
    task.connect({"model_params": model_params})
    
    # Initialize and train model
    print("Training model...")
    model = Prophet(**model_params)
    
    # Add regressors
    for feature in data.columns:
        if feature not in ['ds', 'y']:
            print(f"Adding regressor: {feature}")
            model.add_regressor(feature)
    
    # Add custom seasonalities
    model.add_seasonality(name='rush_hour', period=24, fourier_order=5)
    
    # Fit model
    model.fit(train_data)
    
    # Evaluate on test data
    print("Evaluating model...")
    metrics, forecast = evaluate_model(model, test_data)
    
    # Create and log evaluation plots
    create_evaluation_plots(test_data['y'].values, 
                          forecast['yhat'].values,
                          forecast,
                          task)
    
    # Log metrics
    logger = task.get_logger()
    for metric_name, value in metrics.items():
        if not metric_name.startswith('hour_'):  # Log main metrics
            logger.report_scalar("Main Metrics", metric_name, iteration=0, value=value)
        else:  # Log hourly metrics separately
            logger.report_scalar("Hourly RMSE", metric_name, iteration=0, value=value)
        print(f"{metric_name}: {value:.4f}")
    
    # Save model components
    model_components = {
        'model': model,
        'metrics': metrics,
        'params': model_params,
        'feature_names': list(train_data.columns)
    }
    
    # Save model locally
    joblib.dump(model_components, model_path)
    print(f"Model saved locally to {model_path}")
    
    # Register model in ClearML
    output_model = OutputModel(
        task=task,
        name='TaxiDemandProphet',
        framework='prophet',
        label_enumeration={}
    )
    output_model.update_weights(weights_filename=model_path)
    
    print("Model training completed and logged to ClearML")
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
    Predict taxi demand for the next n hours.
    
    Args:
        input_features: Dictionary containing current feature values
        forecast_hours: Number of hours to forecast ahead
        
    Returns:
        Dictionary containing predictions and confidence intervals
    """
    model_data = load_model()
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Create future dataframe
    future_dates = pd.date_range(
        start=pd.to_datetime(input_features['ds']),
        periods=forecast_hours,
        freq='H'
    )
    
    future = pd.DataFrame({'ds': future_dates})
    
    # Add required regressors
    for feature in feature_names:
        if feature != 'ds' and feature != 'y':
            if feature in input_features:
                future[feature] = input_features[feature]
            else:
                raise ValueError(f"Missing required feature: {feature}")
    
    # Make prediction
    forecast = model.predict(future)
    
    return {
        'timestamps': forecast['ds'].tolist(),
        'predictions': forecast['yhat'].tolist(),
        'lower_bound': forecast['yhat_lower'].tolist(),
        'upper_bound': forecast['yhat_upper'].tolist()
    }

if __name__ == "__main__":
    # Train model on the merged features dataset
    train_and_save_model()
    
    # Test prediction with all required features
    test_features = {
        'ds': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        # Weather features
        'temp': 20.0,
        'feels_like': 19.0,
        'wind_speed': 5.0,
        'rain_1h': 0.0,
        # Distance features
        'avg_distance': 2.5,
        'total_distance': 125.0,
        'unique_pickup_locs': 25,
        # Time-based features
        'hour_of_day': datetime.now().hour,
        'is_weekend': int(datetime.now().weekday() >= 5),
        'is_rush_hour': int(datetime.now().hour in [7,8,9,16,17,18,19])
    }
    
    prediction = predict_demand(test_features)
    print("\nTest prediction for next 24 hours:")
    for i, (ts, pred, lower, upper) in enumerate(zip(
        prediction['timestamps'],
        prediction['predictions'],
        prediction['lower_bound'],
        prediction['upper_bound']
    )):
        print(f"Hour {i+1}: {pred:.2f} [{lower:.2f}, {upper:.2f}]") 