import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import holidays
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def create_evaluation_plots(model, X, y, feature_names, task=None):
    """
    Create and optionally log evaluation plots for the XGBoost model
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
        task: Optional ClearML task object
    """
    # Helper function to handle plot logging
    def log_plot(title, series, fig):
        if task:
            task.get_logger().report_matplotlib_figure(
                title=title,
                series=series,
                figure=fig,
                iteration=0
            )
        plt.close()

    # 1. Feature Importance Plot
    plt.figure(figsize=(12, 6))
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.barh(range(len(importances)), importances['importance'])
    plt.yticks(range(len(importances)), importances['feature'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    log_plot("Feature Importance", "Model Analysis", plt.gcf())

    # 2. Predictions vs Actual
    y_pred = model.predict(X)
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual')
    log_plot("Prediction vs Actual", "Evaluation", plt.gcf())

    # 3. Residual Plot
    residuals = y - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    log_plot("Residuals", "Evaluation", plt.gcf())

    # 4. Residual Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residual Value')
    plt.ylabel('Count')
    plt.title('Residual Distribution')
    log_plot("Residual Distribution", "Evaluation", plt.gcf())

    # Log metrics if ClearML is enabled
    if task:
        metrics_dict = {
            'MSE': mean_squared_error(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MAE': mean_absolute_error(y, y_pred),
            'R2': r2_score(y, y_pred)
        }
        for metric_name, value in metrics_dict.items():
            task.get_logger().report_scalar(
                "Performance Metrics", metric_name, value, 0)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the merged features data for demand prediction
    
    Args:
        df: Merged features DataFrame
        
    Returns:
        Processed DataFrame ready for model training
    """
    # Convert timestamp columns to datetime
    df['hour'] = pd.to_datetime(df['hour'])
    
    # Create time-based features
    df['hour_of_day'] = df['hour'].dt.hour
    df['day_of_week'] = df['hour'].dt.dayofweek
    df['month'] = df['hour'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    # Create time period features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = df['hour_of_day'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    
    # Get US holidays
    us_holidays = holidays.US()
    df['is_holiday'] = df['hour'].dt.date.map(lambda x: x in us_holidays).astype(int)
    
    return df

def train_time_based_model(
    data_path: str, 
    model_path: str = 'models/time_based_model.joblib',
    enable_clearml: bool = True
) -> None:
    """
    Train XGBoost model using only time-based features for long-range forecasting.
    
    Args:
        data_path: Path to the training data CSV
        model_path: Path to save the model
        enable_clearml: Whether to enable ClearML logging
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Initialize ClearML task if enabled
    task = None
    if enable_clearml:
        try:
            from clearml import Task
            task = Task.init(
                project_name='TaxiDemandPrediction',  # Same project as transformer
                task_name=f'XGBoost_TimeBased_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                task_type='training'
            )
        except Exception as e:
            print(f"\nWarning: ClearML initialization failed: {str(e)}")
            print("Continuing without ClearML logging...")
            enable_clearml = False
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Preprocess data to get time-based features
    df = preprocess_data(df)
    
    # Define time-based features to keep
    time_features = [
        'hour_of_day',
        'day_of_week',
        'month',
        'hour_sin',
        'hour_cos',
        'day_sin',
        'day_cos',
        'is_weekend',
        'is_holiday',
        'is_rush_hour',
        'zone_id'  # Keep zone_id as numeric feature
    ]
    
    # Print final feature list
    print("\nFinal input features:")
    for feature in time_features:
        print(f"- {feature}")
    
    # Prepare features and target
    X = df[time_features].copy()
    y = df['demand']  # Use raw demand as target
    
    # Scale features
    scaler = StandardScaler()

    print("\nScaling features...")
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=df.index)
    
    # Model parameters optimized for time-based features
    model_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'eval_metric': ['rmse', 'mae']
    }
    
    # Log hyperparameters if ClearML is enabled
    if task:
        task.connect(model_params)
    
    # Initialize and train model
    model = xgb.XGBRegressor(**model_params)
    print("Training model...")
    model.fit(X_scaled, y, verbose=False)
    print("\nTime-based model training completed")
    
    # Create evaluation plots
    create_evaluation_plots(model, X_scaled, y, time_features, task)
    
    # Save model and metadata
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }, model_path)
    
    print(f"Model saved to {model_path}")
    
    # Close ClearML task if it was initialized
    if task:
        task.close()

if __name__ == "__main__":
    # Train time-based model with optional ClearML
    data_path = 'data_from_2024/taxi_demand_dataset.csv'
    train_time_based_model(data_path, enable_clearml=True)  # Set to False to disable ClearML 