import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('lstm_training')

# Configuration
class Config:
    SEQUENCE_LENGTH = 24  # 24 hours of historical data
    PREDICTION_LENGTH = 24  # Predict next 24 hours
    SAMPLE_SIZE = 10000  # Number of samples to use for testing
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001
    
    # Model architecture
    LSTM_UNITS = [32, 16]
    DROPOUT_RATE = 0.2
    
    # File paths
    DATA_PATH = 'data_from_2024/taxi_demand_dataset.csv'
    MODEL_DIR = 'models'
    SCALER_PATH = 'models/scaler.save'
    MODEL_PATH = 'models/lstm_taxi_model.keras'
    
    # Features
    FEATURES = [
        'temp', 'feels_like', 'wind_speed', 'rain_1h',
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_holiday', 'is_rush_hour',
        'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',  # Lag features
        'demand_rolling_mean_24h', 'demand_rolling_std_24h',  # Rolling statistics
        'demand_change_1h'  # Change features
    ]
    TARGET = 'demand'

def setup_environment():
    """Setup environment, ensure required directories exist"""
    # Set seeds for reproducibility
    np.random.seed(Config.RANDOM_SEED)
    tf.random.set_seed(Config.RANDOM_SEED)
    
    # Create directories if they don't exist
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Set memory growth for GPUs if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            logger.warning(f"GPU memory growth setting failed: {e}")

def load_and_preprocess_data():
    # Load data
    df = pd.read_csv('data_from_2024/taxi_demand_dataset.csv')
    
    # Take a random sample
    df = df.sample(n=Config.SAMPLE_SIZE, random_state=Config.RANDOM_SEED)
    
    # Convert hour to datetime
    df['timestamp'] = pd.to_datetime(df['hour'])
    df.set_index('timestamp', inplace=True)
    
    # Sort by timestamp to maintain temporal order
    df = df.sort_index()
    
    # Select features and target
    data = df[Config.FEATURES + [Config.TARGET]]
    
    # Handle any missing values
    data = data.ffill()  # Using ffill() instead of fillna(method='ffill')
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Save scaler for later use
    joblib.dump(scaler, Config.SCALER_PATH)
    
    return scaled_data, scaler

def create_sequences(data, seq_length, pred_length):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length:i + seq_length + pred_length, -1])  # Last column is target
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(32, input_shape=input_shape, return_sequences=True),  # Reduced from 128
        Dropout(0.2),  # Reduced from 0.3
        LSTM(16),  # Reduced from 64
        Dropout(0.2),  # Reduced from 0.3
        Dense(Config.PREDICTION_LENGTH)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def main():
    # Load and preprocess data
    scaled_data, scaler = load_and_preprocess_data()
    
    # Create sequences
    X, y = create_sequences(scaled_data, Config.SEQUENCE_LENGTH, Config.PREDICTION_LENGTH)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED
    )
    
    # Build and train model
    model = build_model((Config.SEQUENCE_LENGTH, len(Config.FEATURES) + 1))
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE
    )
    
    # Save model
    logger.info(f"Saving model to {Config.MODEL_PATH}")
    model.save(Config.MODEL_PATH)
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae}")
    
    # Print training history
    print("\nTraining History:")
    for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")

if __name__ == "__main__":
    main()
