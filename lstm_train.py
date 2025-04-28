import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import os
import joblib

# Constants
SEQUENCE_LENGTH = 24  # 24 hours of historical data
PREDICTION_LENGTH = 24  # Predict next 24 hours
SAMPLE_SIZE = 10000  # Number of samples to use for testing

# Features to use
FEATURES = [
    'temp', 'feels_like', 'wind_speed', 'rain_1h',
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_holiday', 'is_rush_hour',
    'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',  # Lag features
    'demand_rolling_mean_24h', 'demand_rolling_std_24h',  # Rolling statistics
    'demand_change_1h'  # Change features
]

TARGET = 'demand'

def load_and_preprocess_data():
    # Load data
    df = pd.read_csv('data_from_2024/taxi_demand_dataset.csv')
    
    # Take a random sample
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    
    # Convert hour to datetime
    df['timestamp'] = pd.to_datetime(df['hour'])
    df.set_index('timestamp', inplace=True)
    
    # Sort by timestamp to maintain temporal order
    df = df.sort_index()
    
    # Select features and target
    data = df[FEATURES + [TARGET]]
    
    # Handle any missing values
    data = data.ffill()  # Using ffill() instead of fillna(method='ffill')
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Save scaler for later use
    joblib.dump(scaler, 'scaler.save')
    
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
        Dense(PREDICTION_LENGTH)
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
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH, PREDICTION_LENGTH)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build and train model
    model = build_model((SEQUENCE_LENGTH, len(FEATURES) + 1))
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,  # Reduced from 50
        batch_size=64  # Increased from 32 for faster training
    )
    
    # Save model
    model.save('models/lstm_taxi_model.h5')
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae}")
    
    # Print training history
    print("\nTraining History:")
    for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")

if __name__ == "__main__":
    main()
