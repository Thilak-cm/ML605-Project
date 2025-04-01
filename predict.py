import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
from typing import Dict, Optional
from clearml import Task, Dataset, Model
import tempfile

# Global variable to store the loaded model
_model: Optional[xgb.XGBRegressor] = None

def train_and_save_model(data: pd.DataFrame, model_path: str = 'model.joblib') -> None:
    """
    Train an XGBoost model on the provided data and save it to disk.
    Also logs experiment to ClearML.
    
    Args:
        data: DataFrame containing features and target
        model_path: Path where the model should be saved
    """
    # Initialize ClearML task
    task = Task.init(
        project_name='TaxiDemandPrediction',
        task_name='XGBoostTraining',
        task_type=Task.TaskTypes.TRAINING
    )
    
    # Log dataset info
    task.connect(data)
    
    # Separate features and target
    X = data.drop('demand', axis=1)
    y = data['demand']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Log model parameters
    model_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42
    }
    task.connect(model_params)
    
    # Initialize and train model
    model = xgb.XGBRegressor(**model_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Log metrics
    task.get_logger().report_scalar('MSE', 'Test', mse)
    task.get_logger().report_scalar('RMSE', 'Test', rmse)
    
    print(f"Model MSE: {mse:.2f}")
    print(f"Model RMSE: {rmse:.2f}")
    
    # Save model locally
    joblib.dump(model, model_path)
    print(f"Model saved locally to {model_path}")
    
    # Register model in ClearML
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        joblib.dump(model, tmp.name)
        model = Model(
            task=task,
            name='TaxiDemandXGBoost',
            framework='XGBoost',
            description='Taxi demand prediction model using XGBoost'
        )
        model.upload(tmp.name)
        os.unlink(tmp.name)
    
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
                # Initialize ClearML task
                task = Task.get_task(project_name='TaxiDemandPrediction')
                # Get the latest model
                model = Model.get_model(project_name='TaxiDemandPrediction', name='TaxiDemandXGBoost')
                # Download the model
                model_path = model.download()
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
    # Convert input to DataFrame row
    input_df = pd.DataFrame([input_features])
    
    # Ensure all required features are present
    required_features = [
        'pickup_hour', 'day_of_week', 'zone_id', 
        'temperature', 'rain', 'congestion_level'
    ]
    missing_features = [f for f in required_features if f not in input_features]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Load model and make prediction
    model = load_model()
    prediction = model.predict(input_df)[0]
    
    return float(prediction)

if __name__ == "__main__":
    # Create dummy dataset for testing
    df_dummy = pd.DataFrame({
        "pickup_hour": np.random.randint(0, 24, 100),
        "day_of_week": np.random.randint(0, 7, 100),
        "zone_id": np.random.randint(1, 266, 100),
        "temperature": np.random.uniform(30, 100, 100),
        "rain": np.random.uniform(0, 1, 100),
        "congestion_level": np.random.uniform(0, 1, 100),
        "demand": np.random.randint(0, 150, 100)
    })
    
    # Train and save model
    train_and_save_model(df_dummy)
    
    # Test prediction
    test_features = {
        'pickup_hour': 14,
        'day_of_week': 3,
        'zone_id': 42,
        'temperature': 75.5,
        'rain': 0.2,
        'congestion_level': 0.5
    }
    
    prediction = predict_demand(test_features)
    print(f"Test prediction: {prediction:.2f}")

