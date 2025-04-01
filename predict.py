import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
from typing import Dict, Optional
from clearml import Task, Dataset, OutputModel
import tempfile
from datetime import datetime

# Global variable to store the loaded model
_model: Optional[xgb.XGBRegressor] = None

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the taxi data to create features for demand prediction
    
    Args:
        df: Raw taxi trip data DataFrame
        
    Returns:
        Processed DataFrame with features for demand prediction
    """
    # Extract time-based features
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    
    # Use PULocationID as zone_id
    df['zone_id'] = df['PULocationID']
    
    # Calculate demand (number of pickups per hour per zone)
    demand_df = df.groupby(['pickup_hour', 'day_of_week', 'zone_id']).size().reset_index(name='demand')
    
    # Add congestion level based on congestion surcharge
    congestion_df = df.groupby(['pickup_hour', 'day_of_week', 'zone_id'])['congestion_surcharge'].mean().reset_index()
    demand_df = demand_df.merge(congestion_df, on=['pickup_hour', 'day_of_week', 'zone_id'], how='left')
    demand_df['congestion_level'] = demand_df['congestion_surcharge'] / 2.5  # Normalize to 0-1
    
    return demand_df

def train_and_save_model(data_path: str, model_path: str = 'model.joblib') -> None:
    """
    Train an XGBoost model on the provided data and save it to disk.
    Also logs experiment to ClearML.
    
    Args:
        data_path: Path to the parquet file containing taxi trip data
        model_path: Path where the model should be saved
    """
    # Initialize ClearML task
    task = Task.init(
        project_name='TaxiDemandPrediction',
        task_name='XGBoostTraining',
        task_type='training'
    )
    
    # Load and preprocess data
    print("Loading data...")
    raw_data = pd.read_parquet(data_path)
    data = preprocess_data(raw_data)
    
    # Log dataset info
    task.connect({
        "dataset_size": len(data),
        "num_zones": data['zone_id'].nunique(),
        "date_range": {
            "start": raw_data['tpep_pickup_datetime'].min().strftime('%Y-%m-%d'),
            "end": raw_data['tpep_pickup_datetime'].max().strftime('%Y-%m-%d')
        }
    })
    
    # Separate features and target
    X = data.drop('demand', axis=1)
    y = data['demand']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Model parameters optimized for larger dataset
    model_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 200,  # Increased for better convergence
        'learning_rate': 0.05,  # Reduced for better generalization
        'max_depth': 8,  # Increased for more complex patterns
        'min_child_weight': 3,  # Added to prevent overfitting
        'subsample': 0.8,  # Added for better generalization
        'colsample_bytree': 0.8,  # Added for better generalization
        'random_state': 42,
        'eval_metric': ['rmse']
    }
    
    # Log parameters
    task.connect({"model_params": model_params})
    
    # Initialize and train model
    model = xgb.XGBRegressor(**model_params)
    
    print("Training model...")
    # Train model with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,  # Added early stopping
        verbose=True
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Log final metrics
    logger = task.get_logger()
    logger.report_scalar("Metrics", "MSE", iteration=0, value=mse)
    logger.report_scalar("Metrics", "RMSE", iteration=0, value=rmse)
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    task.connect({"feature_importance": feature_importance.to_dict('records')})
    
    # Log training progress
    validation_scores = model.evals_result()
    for i, rmse_value in enumerate(validation_scores['validation_0']['rmse']):
        logger.report_scalar("Training Progress", "RMSE", iteration=i, value=rmse_value)
    
    print(f"Model MSE: {mse:.2f}")
    print(f"Model RMSE: {rmse:.2f}")
    
    # Save model locally
    joblib.dump(model, model_path)
    print(f"Model saved locally to {model_path}")
    
    # Register model in ClearML
    output_model = OutputModel(
        task=task,
        name='TaxiDemandXGBoost',
        framework='xgboost',
        label_enumeration={}
    )
    output_model.update_weights(weights_filename=model_path)
    
    print("Model registered in ClearML")
    
    # Close the task
    task.close()

def load_model(model_path: str = 'model.joblib') -> xgb.XGBRegressor:
    """
    Load the trained model from disk. Uses a global variable to cache the model.
    If model doesn't exist locally, attempts to download from ClearML.
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Loaded XGBoost model
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
        
        _model = joblib.load(model_path)
    
    return _model

def predict_demand(input_features: Dict) -> float:
    """
    Predict taxi demand based on input features.
    
    Args:
        input_features: Dictionary containing feature values
        
    Returns:
        Predicted demand as a float
    """
    # Ensure all required features are present
    required_features = [
        'pickup_hour', 'day_of_week', 'zone_id', 
        'congestion_surcharge', 'congestion_level'
    ]
    missing_features = [f for f in required_features if f not in input_features]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Identify and notify about extra features
    extra_features = [f for f in input_features if f not in required_features]
    if extra_features:
        print(f"Extra features provided: {extra_features}. These features will not be used.")
    
    # Use only required features from input in the correct order
    input_features = {k: input_features[k] for k in required_features}
    
    # Convert input to DataFrame row with ordered columns
    input_df = pd.DataFrame([input_features], columns=required_features)
    
    # Load model and make prediction
    model = load_model()
    prediction = model.predict(input_df)[0]
    
    return float(prediction)

if __name__ == "__main__":
    # Train model on the full dataset
    data_path = 'data/yellow_tripdata_2025-01.parquet'
    train_and_save_model(data_path)
    
    # Test prediction
    test_features = {
        'pickup_hour': 18,
        'day_of_week': 3,
        'zone_id': 42,
        'congestion_surcharge': 1.25,
        'congestion_level': 0.5
    }
    
    prediction = predict_demand(test_features)
    print(f"Test prediction: {prediction:.2f}")

