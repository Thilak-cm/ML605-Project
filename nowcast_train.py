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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

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
    SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.save')
    MODEL_PATH = os.path.join(MODEL_DIR, 'best_lstm_model.keras')
    PLOT_DIR = os.path.join(MODEL_DIR, 'plots')
    
    # Features
    FEATURES = [
        'temp', 'feels_like', 'wind_speed', 'rain_1h',
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_holiday', 'is_rush_hour',
        'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',  # Lag features
        'demand_rolling_mean_24h', 'demand_rolling_std_24h',  # Rolling statistics
        'demand_change_1h'  # Change features
    ]
    TARGET = 'demand'
    
    # ClearML
    ENABLE_CLEARML = True
    CLEARML_PROJECT = 'TaxiDemandPrediction'
    CLEARML_TASK_NAME = f'LSTM_Training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

def setup_clearml():
    """Initialize ClearML task if enabled and available"""
    if not Config.ENABLE_CLEARML:
        logger.info("ClearML is disabled in config")
        return None

    try:
        from clearml import Task
        task = Task.init(
            project_name=Config.CLEARML_PROJECT,
            task_name=Config.CLEARML_TASK_NAME,
            task_type='training'
        )
        logger.info(f"ClearML initialized: {Config.CLEARML_PROJECT}/{Config.CLEARML_TASK_NAME}")
        
        # Log configuration parameters
        task.connect({
            'SEQUENCE_LENGTH': Config.SEQUENCE_LENGTH,
            'PREDICTION_LENGTH': Config.PREDICTION_LENGTH,
            'BATCH_SIZE': Config.BATCH_SIZE,
            'EPOCHS': Config.EPOCHS,
            'LEARNING_RATE': Config.LEARNING_RATE,
            'LSTM_UNITS': Config.LSTM_UNITS,
            'DROPOUT_RATE': Config.DROPOUT_RATE,
            'TEST_SIZE': Config.TEST_SIZE,
            'SAMPLE_SIZE': Config.SAMPLE_SIZE
        })
        
        return task
    except (ImportError, Exception) as e:
        logger.warning(f"ClearML initialization failed: {str(e)}")
        logger.info("Continuing without ClearML logging")
        return None

def setup_environment():
    """Setup environment, ensure required directories exist"""
    # Set seeds for reproducibility
    np.random.seed(Config.RANDOM_SEED)
    tf.random.set_seed(Config.RANDOM_SEED)
    
    # Create directories if they don't exist
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    
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
    """Load and preprocess the taxi demand dataset"""
    # Load data
    logger.info(f"Loading data from {Config.DATA_PATH}")
    df = pd.read_csv(Config.DATA_PATH)
    
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
    logger.info(f"Data preprocessed: {scaled_data.shape} samples")
    
    return scaled_data, scaler, df

def create_sequences(data, seq_length, pred_length):
    """Create sequences for time series prediction"""
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length:i + seq_length + pred_length, -1])  # Last column is target
    
    X_array = np.array(X)
    y_array = np.array(y)
    
    logger.info(f"Created sequences: X shape {X_array.shape}, y shape {y_array.shape}")
    return X_array, y_array

def build_model(input_shape):
    """Build and compile the LSTM model"""
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
    
    logger.info(f"Model built with input shape {input_shape}")
    model.summary(print_fn=logger.info)
    
    return model

def create_evaluation_plots(history, X_test, y_test, model, task=None):
    """
    Create and save evaluation plots for the LSTM model
    
    Args:
        history: Training history object
        X_test: Test input data
        y_test: Test target data
        model: Trained LSTM model
        task: Optional ClearML task object
    """
    # Create a directory for plots if it doesn't exist
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    
    # Helper function to handle plot logging/saving
    def log_plot(title, series, fig, filename):
        # Save locally
        plt.savefig(os.path.join(Config.PLOT_DIR, filename))
        
        # Log to ClearML if available
        if task:
            try:
                task.get_logger().report_matplotlib_figure(
                    title=title,
                    series=series,
                    figure=fig,
                    iteration=0
                )
            except Exception as e:
                logger.warning(f"Failed to log plot to ClearML: {str(e)}")
        
        plt.close()

    # 1. Training History Plot (Loss curves)
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    log_plot("Loss Curves", "Training", plt.gcf(), "loss_curves.png")
    
    # 2. MAE History Plot
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE During Training')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    log_plot("MAE Curves", "Training", plt.gcf(), "mae_curves.png")
    
    # 3. Predictions vs Actual
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Flatten predictions and actual values for the first timestep
    y_pred_flat = y_pred[:, 0]  # First predicted timestep
    y_test_flat = y_test[:, 0]  # First actual timestep
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_flat, y_pred_flat, alpha=0.5)
    plt.plot([y_test_flat.min(), y_test_flat.max()], 
             [y_test_flat.min(), y_test_flat.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual (First Timestep)')
    plt.grid(True)
    log_plot("Prediction vs Actual", "Evaluation", plt.gcf(), "pred_vs_actual.png")
    
    # 4. Residual Plot
    residuals = y_test_flat - y_pred_flat
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_flat, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot (First Timestep)')
    plt.grid(True)
    log_plot("Residuals", "Evaluation", plt.gcf(), "residuals.png")
    
    # 5. Residual Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residual Value')
    plt.ylabel('Count')
    plt.title('Residual Distribution')
    plt.grid(True)
    log_plot("Residual Distribution", "Evaluation", plt.gcf(), "residual_distribution.png")
    
    # 6. Sample predictions over time
    sample_idx = np.random.choice(len(X_test), size=5, replace=False)
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(sample_idx):
        actual = y_test[idx]
        predicted = y_pred[idx]
        
        plt.subplot(len(sample_idx), 1, i+1)
        plt.plot(range(len(actual)), actual, 'b-', label='Actual')
        plt.plot(range(len(predicted)), predicted, 'r-', label='Predicted')
        plt.title(f'Sample {i+1}: Actual vs Predicted Demand')
        plt.xlabel('Hours')
        plt.ylabel('Normalized Demand')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    log_plot("Time Series Predictions", "Samples", plt.gcf(), "timeseries_samples.png")
    
    # Log metrics if ClearML is enabled
    metrics = {
        'MSE': mean_squared_error(y_test_flat, y_pred_flat),
        'RMSE': np.sqrt(mean_squared_error(y_test_flat, y_pred_flat)),
        'MAE': mean_absolute_error(y_test_flat, y_pred_flat),
        'R2': r2_score(y_test_flat, y_pred_flat)
    }
    
    if task:
        try:
            for metric_name, value in metrics.items():
                task.get_logger().report_scalar(
                    "Performance Metrics", metric_name, value, 0)
        except Exception as e:
            logger.warning(f"Failed to log metrics to ClearML: {str(e)}")
    
    # Log metrics to console as well
    logger.info("Model Evaluation Metrics (First Timestep):")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")

def main():
    """Main function to run the training pipeline"""
    # Set up environment
    setup_environment()
    
    # Initialize ClearML if enabled
    task = setup_clearml()
    
    try:
        # Load and preprocess data
        scaled_data, scaler, raw_df = load_and_preprocess_data()
        
        # Create sequences
        X, y = create_sequences(scaled_data, Config.SEQUENCE_LENGTH, Config.PREDICTION_LENGTH)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED
        )
        
        # Build and train model
        input_shape = (Config.SEQUENCE_LENGTH, len(Config.FEATURES) + 1)
        model = build_model(input_shape)
        
        # Define callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(Config.MODEL_DIR, Config.MODEL_PATH),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Create evaluation plots
        create_evaluation_plots(history, X_test, y_test, model, task)
        
        # Save model
        logger.info(f"Saving model to {Config.MODEL_PATH}")
        model.save(Config.MODEL_PATH)
        
        # Evaluate model
        test_loss, test_mae = model.evaluate(X_test, y_test)
        logger.info(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
        
        # Print training history
        logger.info("\nTraining History:")
        for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            logger.info(f"Epoch {epoch+1}: Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")
            
        # Close ClearML task if it was initialized
        if task:
            try:
                task.close()
                logger.info("ClearML task closed successfully")
            except Exception as e:
                logger.warning(f"Error closing ClearML task: {str(e)}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Attempt to close ClearML task if it was initialized
        if task:
            try:
                task.close()
            except:
                pass
        
        raise

if __name__ == "__main__":
    main()
