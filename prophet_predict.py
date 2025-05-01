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
import argparse
import holidays
from pathlib import Path

# Global variable to store the loaded model
_model: Optional[Prophet] = None

def create_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    forecast_df: pd.DataFrame,
    task: Task
) -> None:
    """
    Create and log evaluation plots to ClearML.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        forecast_df: Prophet forecast DataFrame
        task: ClearML task for logging
    """
    logger = task.get_logger()
    
    # Create time series plot
    plt.figure(figsize=(15, 7))
    plt.plot(y_true, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red')
    plt.title('Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid(True)
    
    # Log to ClearML
    logger.report_matplotlib_figure(
        title='Time Series Plot',
        series='Evaluation',
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    # Create scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Scatter Plot')
    plt.xlabel('Actual Demand')
    plt.ylabel('Predicted Demand')
    plt.grid(True)
    
    # Log to ClearML
    logger.report_matplotlib_figure(
        title='Scatter Plot',
        series='Evaluation',
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    # Create residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(15, 7))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Demand')
    plt.ylabel('Residuals')
    plt.grid(True)
    
    # Log to ClearML
    logger.report_matplotlib_figure(
        title='Residual Plot',
        series='Evaluation',
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()

def preprocess_data(df: pd.DataFrame, zone_id: Optional[int] = None) -> pd.DataFrame:
    """
    Preprocess the data for a univariate Prophet model: only 'ds' and 'y', with outlier capping.
    """
    # Filter by zone if specified
    if zone_id is not None:
        df = df[df['zone_id'] == zone_id].copy()
        print(f"Data points for zone {zone_id}: {len(df)}")
    
    # Aggregate to daily level
    df['date'] = pd.to_datetime(df['hour']).dt.date
    daily_df = df.groupby('date').agg({
        'demand': 'sum',
    }).reset_index()
    
    # Rename columns for Prophet
    prophet_df = daily_df.rename(columns={
        'date': 'ds',
        'demand': 'y'
    })
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    prophet_df = prophet_df[['ds', 'y']]
    
    # Outlier capping
    y_mean = prophet_df['y'].mean()
    y_std = prophet_df['y'].std()
    cap_low = y_mean - 3 * y_std
    cap_high = y_mean + 3 * y_std
    prophet_df['y'] = prophet_df['y'].clip(lower=cap_low, upper=cap_high)
    print(f"\nTotal features used: {prophet_df.columns.tolist()}")
    print("Demand stats after capping:")
    print(prophet_df['y'].describe())
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
    monthly_metrics = {}
    for month in range(1, 13):
        # Get indices for the current month
        month_mask = pd.to_datetime(test_data['ds']).dt.month == month
        if month_mask.any():
            y_true_month = test_data.loc[month_mask, 'y'].values
            y_pred_month = forecast.loc[forecast.index[month_mask], 'yhat'].values
            month_mse = mean_squared_error(y_true_month, y_pred_month)
            monthly_metrics[f'month_{month}_rmse'] = np.sqrt(month_mse)
    
    # Combine all metrics
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        **monthly_metrics
    }
    
    return metrics, forecast

def train_and_save_model(
    data_path: str = 'data_from_2024/taxi_demand_dataset.csv',
    model_path: str = 'prophet_model.joblib',
    zone_id: Optional[int] = None,
    forecast_horizon: int = 31  # Default to predicting March (31 days)
) -> None:
    """
    Train a univariate Prophet model and save it to disk. Log to ClearML.
    Train on data until February 2024, validate on March 2024.
    """
    # Initialize ClearML task
    task = Task.init(
        project_name='TaxiDemandPrediction',
        task_name=f'ProphetTraining_Zone{zone_id if zone_id else "All"}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        task_type='training'
    )
    logger = task.get_logger()
    task.connect_configuration({
        'data_path': data_path,
        'model_path': model_path,
        'zone_id': zone_id,
        'forecast_horizon': forecast_horizon
    })
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    df['hour'] = pd.to_datetime(df['hour'])
    
    # Split data into training (until Feb 2024) and validation (March 2024)
    cutoff_date = pd.Timestamp('2024-03-01')
    df_train = df[df['hour'] < cutoff_date].copy()
    df_val = df[df['hour'].dt.month == 3].copy()
    
    print(f"Training data range: {df_train['hour'].min()} to {df_train['hour'].max()}")
    print(f"Validation data range: {df_val['hour'].min()} to {df_val['hour'].max()}")
    
    print("\nPreprocessing training data...")
    train_df = preprocess_data(df_train, zone_id)
    
    print("\nPreprocessing validation data...")
    val_df = preprocess_data(df_val, zone_id)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")
    
    print("\nTraining univariate Prophet model (growth='flat')...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
        growth='flat'
    )
    model.fit(train_df)
    
    print("\nEvaluating on March 2024 data...")
    metrics, forecast_df = evaluate_model(model, val_df)
    
    print("\nValidation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
        logger.report_scalar(
            title='March 2024 Validation Metrics',
            series=metric_name,
            value=metric_value,
            iteration=0
        )
    
    # Create evaluation plots
    create_evaluation_plots(
        val_df['y'].values,
        forecast_df['yhat'].values,
        forecast_df,
        task
    )
    
    # Save model and metadata
    model_data = {
        'model': model,
        'features': ['ds', 'y'],
        'metrics': metrics,
        'zone_id': zone_id,
        'forecast_horizon': forecast_horizon,
        'training_end': train_df['ds'].max(),
        'validation_period': 'March 2024'
    }
    joblib.dump(model_data, model_path)
    
    output_model = OutputModel(
        task=task,
        name=f'prophet_model_zone_{zone_id if zone_id else "all"}'
    )
    output_model.update_weights(model_path)
    
    # Generate validation period summary
    val_summary = {
        'mean_actual_demand': val_df['y'].mean(),
        'mean_predicted_demand': forecast_df['yhat'].mean(),
        'max_actual_demand': val_df['y'].max(),
        'max_predicted_demand': forecast_df['yhat'].max(),
        'min_actual_demand': val_df['y'].min(),
        'min_predicted_demand': forecast_df['yhat'].min()
    }
    
    print("\nMarch 2024 Validation Summary:")
    for key, value in val_summary.items():
        print(f"{key}: {value:.2f}")
    
    print("\nTraining complete!")
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

def predict_demand(
    input_features: Dict[str, any],
    forecast_horizon: int = 365,
    zone_id: Optional[int] = None
) -> Dict[str, List[float]]:
    """
    Predict taxi demand for the next N days using the trained Prophet model.
    
    Args:
        input_features: Dictionary containing feature values for prediction
        forecast_horizon: Number of days to forecast
        zone_id: Optional zone ID to predict for
        
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
    
    # Create future dataframe for prediction
    future_dates = pd.date_range(
        start=input_features['ds'],
        periods=forecast_horizon,
        freq='D'
    )
    
    # Create prediction dataframe with all required features
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Add time-based features
    future_df['day_of_week'] = future_df['ds'].dt.dayofweek
    future_df['month'] = future_df['ds'].dt.month
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
    future_df['is_summer'] = future_df['month'].isin([6, 7, 8]).astype(int)
    future_df['is_winter'] = future_df['month'].isin([12, 1, 2]).astype(int)
    
    # Add holiday information
    us_holidays = holidays.US()
    future_df['is_holiday'] = future_df['ds'].dt.date.apply(lambda x: x in us_holidays).astype(int)
    
    # Add other features from input_features
    for feature in required_features:
        if feature not in future_df.columns:
            future_df[feature] = input_features.get(feature, 0)
    
    # Make prediction
    forecast = _model.predict(future_df)
    
    # Return predictions
    return {
        'timestamps': future_dates.strftime('%Y-%m-%d').tolist(),
        'demand': forecast['yhat'].tolist(),
        'demand_lower': forecast['yhat_lower'].tolist(),
        'demand_upper': forecast['yhat_upper'].tolist()
    }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and validate Prophet model for taxi demand prediction')
    parser.add_argument('--data_path', type=str, default='data_from_2024/taxi_demand_dataset.csv',
                      help='Path to the input data CSV file')
    parser.add_argument('--model_path', type=str, default='prophet_model.joblib',
                      help='Path to save the trained model')
    parser.add_argument('--zone_id', type=int, default=None,
                      help='Zone ID to train/predict for (optional)')
    parser.add_argument('--forecast_horizon', type=int, default=31,
                      help='Number of days to forecast (default: 31 for March)')
    
    args = parser.parse_args()
    
    train_and_save_model(
        data_path=args.data_path,
        model_path=args.model_path,
        zone_id=args.zone_id,
        forecast_horizon=args.forecast_horizon
    ) 