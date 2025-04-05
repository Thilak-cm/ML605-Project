import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from typing import Dict, Optional
from clearml import Task, Dataset, OutputModel
import tempfile
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import shap

# Global variable to store the loaded model
_model: Optional[xgb.XGBRegressor] = None

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
    df['dt_iso'] = pd.to_datetime(df['dt_iso'])
    
    # Identify numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # Handle duplicate timestamps
    # For numeric columns, take the mean
    # For non-numeric columns, take the first value
    numeric_agg = df.groupby('hour')[numeric_cols].mean()
    non_numeric_agg = df.groupby('hour')[non_numeric_cols].first()
    
    # Combine the aggregations
    df = pd.concat([numeric_agg, non_numeric_agg], axis=1)
    
    # Create time-based features
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    # Log transform the demand and distance-related features
    for col in ['demand', 'avg_distance', 'total_distance']:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    
    for col in [c for c in df.columns if 'lag' in c or 'rolling_mean' in c]:
        df[col] = np.log1p(df[col])
    
    # Drop redundant columns
    df = df.drop(['dt_iso', 'weather_description'], axis=1, errors='ignore')
    
    return df

def create_evaluation_plots(y_true: np.ndarray, y_pred: np.ndarray, 
                          feature_importance: pd.DataFrame, 
                          task: Task,
                          model: xgb.XGBRegressor = None,
                          X_test: pd.DataFrame = None,
                          dates: pd.DatetimeIndex = None) -> None:
    """
    Create and log evaluation plots to ClearML
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        feature_importance: DataFrame with feature importance
        task: ClearML task object
        model: Trained XGBoost model (optional)
        X_test: Test features DataFrame (optional)
        dates: DatetimeIndex for time-based analysis (optional)
    """
    logger = task.get_logger()
    
    # 1. Basic Evaluation Plots
    
    # Create residual plot
    plt.figure(figsize=(12, 6))
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    logger.report_matplotlib_figure(
        title="Residual Plot",
        series="Basic Evaluation",
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    # Create actual vs predicted plot
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    logger.report_matplotlib_figure(
        title="Actual vs Predicted",
        series="Basic Evaluation",
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    # 2. Feature Analysis
    
    # Create feature importance plot with improved styling
    plt.figure(figsize=(12, 8))
    importance_plot = sns.barplot(
        x='importance',
        y='feature',
        data=feature_importance.sort_values('importance', ascending=False).head(20),
        palette='viridis'
    )
    plt.title('Top 20 Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    
    # Add value labels to the bars
    for i in importance_plot.containers:
        importance_plot.bar_label(i, fmt='%.3f')
    
    logger.report_matplotlib_figure(
        title="Feature Importance",
        series="Feature Analysis",
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    # 3. Error Analysis
    
    # Error distribution plot
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, kde=True, bins=50)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    logger.report_matplotlib_figure(
        title="Error Distribution",
        series="Error Analysis",
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    # Q-Q plot for residuals
    plt.figure(figsize=(10, 6))
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    logger.report_matplotlib_figure(
        title="Q-Q Plot",
        series="Error Analysis",
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    if dates is not None:
        # 4. Time-based Analysis
        
        # Error by hour of day
        hourly_errors = pd.DataFrame({
            'hour': pd.DatetimeIndex(dates).hour,
            'abs_error': np.abs(residuals),
            'error': residuals
        })
        
        # Mean absolute error by hour
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='hour', y='abs_error', data=hourly_errors)
        plt.xlabel('Hour of Day')
        plt.ylabel('Absolute Error')
        plt.title('Error Distribution by Hour of Day')
        logger.report_matplotlib_figure(
            title="Hourly Error Distribution",
            series="Time Analysis",
            figure=plt.gcf(),
            iteration=0
        )
        plt.close()
        
        # Error patterns over time
        plt.figure(figsize=(15, 6))
        plt.scatter(dates, residuals, alpha=0.5, s=10)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Error')
        plt.title('Error Patterns Over Time')
        logger.report_matplotlib_figure(
            title="Temporal Error Patterns",
            series="Time Analysis",
            figure=plt.gcf(),
            iteration=0
        )
        plt.close()
        
        # Weekly patterns
        weekly_errors = pd.DataFrame({
            'day': pd.DatetimeIndex(dates).dayofweek,
            'abs_error': np.abs(residuals)
        })
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='day', y='abs_error', data=weekly_errors)
        plt.xlabel('Day of Week (0=Monday)')
        plt.ylabel('Absolute Error')
        plt.title('Error Distribution by Day of Week')
        logger.report_matplotlib_figure(
            title="Weekly Error Patterns",
            series="Time Analysis",
            figure=plt.gcf(),
            iteration=0
        )
        plt.close()
    
    if model is not None and X_test is not None:
        # 5. SHAP Analysis
        try:
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test, show=False)
            plt.title('SHAP Feature Impact Overview')
            logger.report_matplotlib_figure(
                title="SHAP Summary",
                series="SHAP Analysis",
                figure=plt.gcf(),
                iteration=0
            )
            plt.close()
            
            # SHAP dependence plots for top 3 features
            top_features = feature_importance.nlargest(3, 'importance')['feature'].values
            for feature in top_features:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    feature, shap_values, X_test,
                    show=False
                )
                plt.title(f'SHAP Dependence Plot: {feature}')
                logger.report_matplotlib_figure(
                    title=f"SHAP Dependence - {feature}",
                    series="SHAP Analysis",
                    figure=plt.gcf(),
                    iteration=0
                )
                plt.close()
            
            # SHAP Interaction plots for top feature
            top_feature = feature_importance.nlargest(1, 'importance')['feature'].values[0]
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                top_feature, shap_values, X_test,
                interaction_index='auto',
                show=False
            )
            plt.title(f'SHAP Interaction Plot: {top_feature}')
            logger.report_matplotlib_figure(
                title=f"SHAP Interaction - {top_feature}",
                series="SHAP Analysis",
                figure=plt.gcf(),
                iteration=0
            )
            plt.close()
            
        except Exception as e:
            print(f"Could not create SHAP plots: {str(e)}")
    
    # 6. Prediction Range Analysis
    
    # Error vs Prediction Magnitude
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, np.abs(residuals), alpha=0.5)
    plt.xlabel('Predicted Value')
    plt.ylabel('Absolute Error')
    plt.title('Error Magnitude vs Prediction Value')
    logger.report_matplotlib_figure(
        title="Error vs Prediction",
        series="Prediction Analysis",
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    # Prediction vs Actual with confidence bands
    plt.figure(figsize=(12, 6))
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.5})
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual with Confidence Bands')
    logger.report_matplotlib_figure(
        title="Regression Plot",
        series="Prediction Analysis",
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()

def train_and_save_model(data_path: str, model_path: str = 'model.joblib') -> None:
    """
    Train XGBoost model and save it along with preprocessing objects.
    
    Args:
        data_path: Path to the training data CSV
        model_path: Path to save the model
    """
    # Initialize ClearML task
    task = Task.init(
        project_name='TaxiDemandPrediction',
        task_name='XGBoostTraining_Normalized',
        task_type='training'
    )
    logger = task.get_logger()
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Convert datetime column
    if 'hour' in df.columns:
        df['hour'] = pd.to_datetime(df['hour'])
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Define features to exclude (in addition to datetime and demand)
    exclude_features = ['demand', 'trip_count', 'unique_vendors', 'weather_id']
    
    # Prepare features and target
    X = df.select_dtypes(exclude=['datetime64'])
    X = X.drop([col for col in exclude_features if col in X.columns], axis=1)
    y = df['demand']  # Use normalized demand as target
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Print feature information
    print("\nFeature Information:")
    print(f"Total features: {len(feature_names)}")
    print("Features:", feature_names)
    
    # Create train/test splits
    n_splits = 3
    test_size = len(X) // 5  # 20% for testing
    
    print(f"\nUsing {n_splits} splits with test_size={test_size} for {len(X)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=df.index)
    
    # Use time series split for validation with adjusted parameters
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    # Improved model parameters
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
    
    # Log parameters
    task.connect({
        "model_params": model_params,
        "cv_params": {
            "n_splits": n_splits,
            "test_size": test_size,
            "total_samples": len(X)
        }
    })
    
    # Initialize model
    model = xgb.XGBRegressor(**model_params)
    
    # Perform time series cross-validation
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        print(f"Training fold {fold + 1}/{n_splits}")
        X_train = X_scaled.iloc[train_idx]
        X_val = X_scaled.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Make predictions and inverse transform for proper metrics
        y_pred = model.predict(X_val)
        
        # Convert back from log scale for interpretable metrics
        y_val_orig = np.expm1(y_val)
        y_pred_orig = np.expm1(y_pred)
        
        mse = mean_squared_error(y_val_orig, y_pred_orig)
        mae = mean_absolute_error(y_val_orig, y_pred_orig)
        r2 = r2_score(y_val_orig, y_pred_orig)
        
        cv_scores.append({
            'fold': fold + 1,
            'mse': mse,
            'mae': mae,
            'r2': r2
        })
        
        # Log metrics for each fold
        logger.report_scalar("CV Metrics", "MSE", iteration=fold, value=mse)
        logger.report_scalar("CV Metrics", "MAE", iteration=fold, value=mae)
        logger.report_scalar("CV Metrics", "R2", iteration=fold, value=r2)
        
        print(f"Fold {fold + 1} metrics - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")
    
    print("\nTraining final model on full dataset...")
    
    # Train final model on full dataset
    model.fit(X_scaled, y, verbose=False)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    
    # Create and log evaluation plots
    y_pred = model.predict(X_scaled)
    
    # Convert predictions back from log scale
    y_orig = np.expm1(y)
    y_pred_orig = np.expm1(y_pred)
    
    create_evaluation_plots(y_orig, y_pred_orig, feature_importance, task, model, X_scaled, df.index)
    
    # Log final metrics
    final_metrics = {
        'mean_mse': np.mean([s['mse'] for s in cv_scores]),
        'mean_mae': np.mean([s['mae'] for s in cv_scores]),
        'mean_r2': np.mean([s['r2'] for s in cv_scores])
    }
    task.connect({"final_metrics": final_metrics})
    
    print("\nFinal average metrics:")
    print(f"Mean MSE: {final_metrics['mean_mse']:.2f}")
    print(f"Mean MAE: {final_metrics['mean_mae']:.2f}")
    print(f"Mean R2: {final_metrics['mean_r2']:.3f}")
    
    # Save model, scaler, and preprocessing info
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'is_log_transformed': True  # Flag to indicate log transformation
    }, model_path)
    
    # Register model in ClearML
    output_model = OutputModel(
        task=task,
        name='TaxiDemandXGBoost',
        framework='xgboost',
        label_enumeration={}
    )
    output_model.update_weights(weights_filename=model_path)
    
    print("\nModel training completed and logged to ClearML")
    task.close()

def load_model(model_path: str = 'model.joblib') -> Dict:
    """
    Load the trained model and scaler from disk.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Dictionary containing model, scaler, and feature names
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    return joblib.load(model_path)

def predict_demand(input_features: Dict) -> float:
    """
    Predict taxi demand based on input features.
    
    Args:
        input_features: Dictionary containing feature values
        
    Returns:
        Predicted demand as a float
    """
    model_data = load_model()
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    is_log_transformed = model_data.get('is_log_transformed', False)
    
    # Create DataFrame with input features
    input_df = pd.DataFrame([input_features])
    
    # Ensure all required features are present
    missing_features = set(feature_names) - set(input_df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Reorder columns to match training data
    input_df = input_df[feature_names]
    
    # Log transform input features if needed
    if is_log_transformed:
        for col in input_df.columns:
            if 'demand' in col or 'distance' in col or 'lag' in col or 'rolling_mean' in col:
                input_df[col] = np.log1p(input_df[col])
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Inverse transform if log transformed
    if is_log_transformed:
        prediction = np.expm1(prediction)
    
    return float(prediction)

if __name__ == "__main__":
    # Train model on the merged features dataset
    data_path = 'data_from_2024/merged_features.csv'
    train_and_save_model(data_path)
    
    # Test prediction with sample features
    test_features = {
        'hour_of_day': 18,
        'day_of_week': 3,
        'month': 6,
        'temp': 25.0,
        'feels_like': 26.0,
        'temp_min': 23.0,
        'temp_max': 27.0,
        'wind_speed': 5.0,
        'rain_1h': 0.0,
        'is_weekend': 0,
        'is_rush_hour': 1,
        'is_holiday': 0,
        'day_of_month': 15,
        'week_of_year': 24,
        'period_night': 0,
        'period_morning': 0,
        'period_afternoon': 0,
        'period_evening': 1,
        'weather_Clear': 1,
        'weather_Clouds': 0,
        'weather_Drizzle': 0,
        'weather_Fog': 0,
        'weather_Haze': 0,
        'weather_Mist': 0,
        'weather_Rain': 0,
        'weather_Snow': 0,
        'weather_Squall': 0,
        'weather_Thunderstorm': 0,
        'hour_sin': np.sin(2 * np.pi * 18/24),
        'hour_cos': np.cos(2 * np.pi * 18/24),
        'day_sin': np.sin(2 * np.pi * 3/7),
        'day_cos': np.cos(2 * np.pi * 3/7),
        # Lagged features (normalized to 0-100 scale)
        'demand_lag_1h': 50.0,
        'demand_lag_3h': 45.0,
        'demand_lag_24h': 40.0,
        'demand_lag_168h': 35.0,
        'avg_distance_lag_1h': 2.5,
        'avg_distance_lag_3h': 2.4,
        'avg_distance_lag_24h': 2.3,
        'avg_distance_lag_168h': 2.2,
        'demand_rolling_mean_3h': 48.0,
        'demand_rolling_mean_6h': 46.0,
        'demand_rolling_mean_12h': 44.0,
        'demand_rolling_mean_24h': 42.0,
        'avg_distance_rolling_mean_3h': 2.45,
        'avg_distance_rolling_mean_6h': 2.4,
        'avg_distance_rolling_mean_12h': 2.35,
        'avg_distance_rolling_mean_24h': 2.3,
        'avg_distance': 2.5,
        'total_distance': 125.0,
        'unique_pickup_locs': 25
    }
    
    prediction = predict_demand(test_features)
    print(f"Test prediction: {prediction:.2f}")

