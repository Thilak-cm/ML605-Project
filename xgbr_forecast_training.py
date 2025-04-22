import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import holidays

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

def train_time_based_model(data_path: str, model_path: str = 'time_based_model.joblib') -> None:
    """
    Train XGBoost model using only time-based features for long-range forecasting.
    
    Args:
        data_path: Path to the training data CSV
        model_path: Path to save the model
    """
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
    
    # Initialize and train model
    model = xgb.XGBRegressor(**model_params)
    model.fit(X_scaled, y, verbose=False)
    
    # Save model and metadata
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }, model_path)
    
    print("\nTime-based model training completed")

if __name__ == "__main__":
    # Train time-based model
    data_path = 'data_from_2024/merged_features.csv'
    train_time_based_model(data_path) 