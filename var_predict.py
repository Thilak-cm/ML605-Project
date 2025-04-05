import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from clearml import Task, OutputModel
import joblib
from datetime import datetime, timedelta

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Preprocess the data for VAR model.
    
    Args:
        df: Input DataFrame with features
        
    Returns:
        Tuple of (processed DataFrame, scaler)
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Convert hour to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(data['hour']):
        data['hour'] = pd.to_datetime(data['hour'])
    
    # Set hour as index
    data.set_index('hour', inplace=True)
    
    # Select features for VAR model
    features = [
        'demand',
        'temp',
        'feels_like',
        'wind_speed',
        'rain_1h',
        'avg_distance',
        'total_distance',
        'unique_pickup_locs'
    ]
    
    # Keep only selected features
    data = data[features]
    
    # Handle missing values if any
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    
    return scaled_df, scaler

def create_evaluation_plots(y_true: np.ndarray, y_pred: np.ndarray, 
                          dates: pd.DatetimeIndex,
                          task: Task) -> None:
    """
    Create and log evaluation plots to ClearML
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        dates: DatetimeIndex for the predictions
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
    plt.plot(dates, y_true, label='Actual', alpha=0.5)
    plt.plot(dates, y_pred, label='Predicted', alpha=0.5)
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

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def train_and_save_model(data_path: str = 'data_from_2024/merged_features.csv',
                        model_path: str = 'var_model.joblib') -> None:
    """
    Train VAR model and save it along with the scaler
    
    Args:
        data_path: Path to input data
        model_path: Path to save model
    """
    # Initialize ClearML task
    task = Task.init(
        project_name='TaxiDemandPrediction',
        task_name='VARTraining',
        task_type='training'
    )
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Preprocess data
    print("Preprocessing data...")
    processed_data, scaler = preprocess_data(df)
    
    # Split into train and test (last 20% for testing)
    train_size = int(len(processed_data) * 0.8)
    train_data = processed_data[:train_size]
    test_data = processed_data[train_size:]
    
    # Log dataset info
    task.connect({
        "dataset_info": {
            "train_size": len(train_data),
            "test_size": len(test_data),
            "train_date_range": {
                "start": train_data.index.min().strftime('%Y-%m-%d %H:%M'),
                "end": train_data.index.max().strftime('%Y-%m-%d %H:%M')
            },
            "test_date_range": {
                "start": test_data.index.min().strftime('%Y-%m-%d %H:%M'),
                "end": test_data.index.max().strftime('%Y-%m-%d %H:%M')
            }
        }
    })
    
    # Train VAR model
    print("Training model...")
    model = VAR(train_data)
    
    # Find optimal order using AIC
    max_lags = 24  # Test up to 24 lags (24 hours)
    results = []
    for i in range(1, max_lags + 1):
        result = model.fit(i)
        results.append([i, result.aic])
    
    # Select best order
    best_order = min(results, key=lambda x: x[1])[0]
    print(f"Best VAR order: {best_order}")
    
    # Fit model with best order
    model_fitted = model.fit(best_order)
    
    # Make predictions on test set
    print("Evaluating model...")
    lag_order = model_fitted.k_ar
    test_input = train_data[-lag_order:]
    test_pred = []
    
    for i in range(len(test_data)):
        forecast = model_fitted.forecast(test_input.values, steps=1)
        test_pred.append(forecast[0])
        test_input = pd.DataFrame(
            np.vstack([test_input.values[1:], forecast]),
            columns=test_input.columns,
            index=test_input.index[1:].append(pd.DatetimeIndex([test_input.index[-1] + pd.Timedelta(hours=1)]))
        )
    
    test_pred = np.array(test_pred)
    
    # Transform predictions back to original scale for demand
    demand_idx = processed_data.columns.get_loc('demand')
    y_true = scaler.inverse_transform(test_data.values)[:, demand_idx]
    y_pred = scaler.inverse_transform(test_pred)[:, demand_idx]
    
    # Calculate metrics
    metrics = evaluate_model(y_true, y_pred)
    
    # Create and log evaluation plots
    create_evaluation_plots(y_true, y_pred, test_data.index, task)
    
    # Log metrics
    logger = task.get_logger()
    for metric_name, value in metrics.items():
        logger.report_scalar("Metrics", metric_name, iteration=0, value=value)
        print(f"{metric_name}: {value:.4f}")
    
    # Save model components
    model_components = {
        'model': model_fitted,
        'scaler': scaler,
        'features': processed_data.columns.tolist(),
        'metrics': metrics,
        'best_order': best_order
    }
    
    # Save model locally
    joblib.dump(model_components, model_path)
    print(f"Model saved locally to {model_path}")
    
    # Register model in ClearML
    output_model = OutputModel(
        task=task,
        name='TaxiDemandVAR',
        framework='statsmodels',
        label_enumeration={}
    )
    output_model.update_weights(weights_filename=model_path)
    
    print("Model training completed and logged to ClearML")
    task.close()

def predict_demand(input_features: Dict[str, any], forecast_hours: int = 24) -> Dict[str, List[float]]:
    """
    Predict taxi demand for the next n hours
    
    Args:
        input_features: Dictionary containing current feature values
        forecast_hours: Number of hours to forecast ahead
        
    Returns:
        Dictionary containing predictions
    """
    # Load model components
    model_components = joblib.load('var_model.joblib')
    model = model_components['model']
    scaler = model_components['scaler']
    features = model_components['features']
    
    # Prepare input data
    input_df = pd.DataFrame([input_features])
    input_scaled = scaler.transform(input_df[features])
    
    # Make forecast
    forecast = model.forecast(input_scaled, steps=forecast_hours)
    
    # Transform predictions back to original scale
    forecast_original = scaler.inverse_transform(forecast)
    demand_idx = features.index('demand')
    
    # Prepare timestamps
    start_time = pd.to_datetime(input_features.get('hour', datetime.now()))
    timestamps = [start_time + timedelta(hours=i) for i in range(forecast_hours)]
    
    return {
        'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
        'predictions': forecast_original[:, demand_idx].tolist()
    }

if __name__ == "__main__":
    # Train model
    train_and_save_model()
    
    # Test prediction
    test_features = {
        'hour': datetime.now(),
        'demand': 100,  # Current demand
        'temp': 20.0,
        'feels_like': 19.0,
        'wind_speed': 5.0,
        'rain_1h': 0.0,
        'avg_distance': 2.5,
        'total_distance': 125.0,
        'unique_pickup_locs': 25
    }
    
    prediction = predict_demand(test_features)
    print("\nTest prediction for next 24 hours:")
    for ts, pred in zip(prediction['timestamps'], prediction['predictions']):
        print(f"{ts}: {pred:.2f}") 