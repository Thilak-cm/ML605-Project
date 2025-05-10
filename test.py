import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from clearml import Task
import joblib
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import optuna
from optuna.trial import Trial
import optuna.visualization as vis
import plotly.graph_objects as go
import psutil
import gc
import signal
import logging
from pathlib import Path

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'transformer_training_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    # Create logger
    logger = logging.getLogger('transformer_training')
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

# Initialize logger
logger = setup_logging()

class SimpleTimeSeriesTransformer(nn.Module):
    """
    Simplified Transformer model for time series forecasting.
    Uses minimal layers and parameters for quick training and iteration.
    """
    def __init__(
        self,
        d_model: int = 8,
        n_heads: int = 2,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        dropout: float = 0.1,
        input_seq_len: int = 24,
        output_seq_len: int = 24,
        n_features: int = 1,
        n_static_features: int = 1
    ):
        super().__init__()
        
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.n_features = n_features
        
        # Input projections
        self.feature_projection = nn.Linear(n_features, d_model)
        self.static_projection = nn.Linear(n_static_features, d_model)
        
        # Simplified positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, input_seq_len + output_seq_len, d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
    
    def forward(self, src: torch.Tensor, tgt_features: torch.Tensor, static_features: torch.Tensor) -> torch.Tensor:
        # Project input features
        src = self.feature_projection(src)  # [batch, input_seq_len, d_model]
        tgt = self.feature_projection(tgt_features)  # [batch, output_seq_len, d_model]
        
        # Project and expand static features
        static = self.static_projection(static_features)  # [batch, d_model]
        static = static.unsqueeze(1).expand(-1, src.size(1), -1)  # [batch, seq_len, d_model]
        
        # Add static features and positional encoding
        src = src + static + self.pos_encoder[:, :src.size(1)]
        tgt = tgt + static[:, :tgt.size(1)] + self.pos_encoder[:, :tgt.size(1)]
        
        # Create mask for decoder
        tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer forward pass
        memory = self.encoder(src)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # Reshape for output projection
        batch_size, seq_len, d_model = output.shape
        output = output.reshape(-1, d_model)  # [batch*seq_len, d_model]
        output = self.output_projection(output)  # [batch*seq_len, 1]
        output = output.reshape(batch_size, seq_len, 1)  # [batch, seq_len, 1]
        
        return output

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class TimeSeriesDataset(Dataset):
    """Dataset class for time series data with multiple features."""
    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        static_columns: List[str],
        target_column: str = 'demand',
        input_seq_len: int = 24,
        output_seq_len: int = 24,
        stride: int = 1
    ):
        self.feature_columns = feature_columns
        self.static_columns = static_columns
        self.target_column = target_column
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.stride = stride
        
        # Group by zone_id and convert to list of sequences
        self.sequences = []
        
        print(f"\nCreating sequences from {len(data)} total records")
        print(f"Unique zones: {data['zone_id'].nunique()}")
        print(f"Features used: {feature_columns}")
        
        for zone_id, group in data.groupby('zone_id'):
            if len(group) < input_seq_len + output_seq_len:
                print(f"Skipping zone {zone_id}: insufficient data (only {len(group)} records)")
                continue
                
            # Sort by hour to ensure temporal ordering
            group = group.sort_values('hour')
            
            # Create sequences with overlap
            n_sequences = (len(group) - input_seq_len - output_seq_len) // stride + 1
            
            for i in range(0, len(group) - input_seq_len - output_seq_len + 1, stride):
                # Input features
                features = torch.FloatTensor(group[feature_columns].iloc[i:i + input_seq_len].values)
                
                # Target sequence (using all features for the target timeframe)
                target_idx = slice(i + input_seq_len, i + input_seq_len + output_seq_len)
                target_features = torch.FloatTensor(group[feature_columns].iloc[target_idx].values)
                
                # Static features (zone information)
                static = torch.FloatTensor(group[static_columns].iloc[0].values)
                
                self.sequences.append({
                    'features': features,  # [input_seq_len, n_features]
                    'target_features': target_features,  # [output_seq_len, n_features]
                    'target': torch.FloatTensor(group[target_column].iloc[target_idx].values).unsqueeze(-1),  # [output_seq_len, 1]
                    'static': static  # [n_static_features]
                })
            
            if len(group) > 1000:  # Only print for zones with significant data
                print(f"Zone {zone_id}: created {n_sequences} sequences from {len(group)} records")
        
        print(f"\nTotal sequences created: {len(self.sequences)}")
        if len(self.sequences) == 0:
            raise ValueError("No valid sequences could be created! Check your data and sequence lengths.")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        return (
            seq['features'],  # [input_seq_len, n_features]
            seq['target_features'],  # [output_seq_len, n_features]
            seq['target'],  # [output_seq_len, 1]
            seq['static']  # [n_static_features]
        )

def create_evaluation_plots(model, y_true, y_pred, history, task=None, dates=None):
    """
    Create comprehensive evaluation plots and optionally log them to ClearML
    
    Args:
        model: Trained transformer model
        y_true: True values
        y_pred: Predicted values
        history: Training history dictionary
        task: Optional ClearML task object. If None, plots are only displayed
        dates: Optional datetime index for time-based plots
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
    
    # Reshape arrays to 2D for plotting
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    residuals = y_true_flat - y_pred_flat
    
    # 1. Training History Plot
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    log_plot("Training History", "Training", plt.gcf())
    
    # 2. Prediction vs Actual Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.5)
    plt.plot([y_true_flat.min(), y_true_flat.max()], [y_true_flat.min(), y_true_flat.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual')
    log_plot("Prediction vs Actual", "Evaluation", plt.gcf())
    
    # 3. Residual Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred_flat, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    log_plot("Residuals", "Evaluation", plt.gcf())
    
    # 4. Residual Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residual Value')
    plt.ylabel('Count')
    plt.title('Residual Distribution')
    log_plot("Residual Distribution", "Evaluation", plt.gcf())
    
    # 5. Error Metrics Over Time
    if dates is not None:
        # Flatten and align dates with residuals
        flat_dates = []
        for date_series in dates:
            flat_dates.extend(date_series)
        flat_dates = flat_dates[:len(residuals)]
        
        plt.figure(figsize=(15, 6))
        plt.plot(flat_dates, np.abs(residuals), alpha=0.5)
        plt.xlabel('Date')
        plt.ylabel('Absolute Error')
        plt.title('Absolute Error Over Time')
        plt.xticks(rotation=45)
        log_plot("Error Over Time", "Time Analysis", plt.gcf())
        
        # 6. Hourly Error Patterns
        hourly_errors = pd.DataFrame({
            'hour': pd.DatetimeIndex(flat_dates).hour,
            'abs_error': np.abs(residuals)
        })
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='hour', y='abs_error', data=hourly_errors)
        plt.xlabel('Hour of Day')
        plt.ylabel('Absolute Error')
        plt.title('Error Distribution by Hour')
        log_plot("Hourly Error Patterns", "Time Analysis", plt.gcf())
    
    # 7. Prediction Intervals (if available)
    if hasattr(model, 'predict_interval'):
        try:
            lower, upper = model.predict_interval(y_pred)
            plt.figure(figsize=(15, 6))
            plt.fill_between(range(len(y_pred_flat)), lower.flatten(), upper.flatten(), alpha=0.3, label='95% CI')
            plt.plot(y_true_flat, label='Actual', alpha=0.7)
            plt.plot(y_pred_flat, label='Predicted', alpha=0.7)
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.title('Predictions with Confidence Intervals')
            plt.legend()
            log_plot("Prediction Intervals", "Evaluation", plt.gcf())
        except:
            pass

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 5,
    learning_rate: float = 0.001,
    patience: int = 5
) -> Dict[str, List[float]]:
    """Training loop with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        
        train_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Train]')
        for batch_x, batch_tgt_features, batch_y, batch_static in train_iterator:
            batch_x = batch_x.to(device)
            batch_tgt_features = batch_tgt_features.to(device)
            batch_y = batch_y.to(device)
            batch_static = batch_static.to(device)
            
            optimizer.zero_grad()
            y_pred = model(batch_x, batch_tgt_features, batch_static)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            train_iterator.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            val_iterator = tqdm(val_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Val]')
            for batch_x, batch_tgt_features, batch_y, batch_static in val_iterator:
                batch_x = batch_x.to(device)
                batch_tgt_features = batch_tgt_features.to(device)
                batch_y = batch_y.to(device)
                batch_static = batch_static.to(device)
                
                y_pred = model(batch_x, batch_tgt_features, batch_static)
                val_loss = criterion(y_pred, batch_y)
                val_losses.append(val_loss.item())
                
                val_iterator.set_postfix({'loss': f'{val_loss.item():.4f}'})
        
        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping with model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history

def calculate_original_scale_metrics(y_true, y_pred, scaler, task=None):
    """
    Calculate performance metrics in the original scale of values.
    
    Args:
        y_true: True values in scaled form
        y_pred: Predicted values in scaled form
        scaler: The StandardScaler used to scale the data
        task: Optional ClearML task object for logging
    """
    # Ensure inputs are 2D arrays
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    
    # Create dummy arrays with the same shape as our target
    dummy_true = np.zeros((len(y_true), scaler.n_features_in_))
    dummy_pred = np.zeros((len(y_pred), scaler.n_features_in_))
    
    # Put our values in the first column (assuming demand is the first feature)
    dummy_true[:, 0] = y_true.flatten()
    dummy_pred[:, 0] = y_pred.flatten()
    
    # Inverse transform to get original scale
    true_original = scaler.inverse_transform(dummy_true)[:, 0]
    pred_original = scaler.inverse_transform(dummy_pred)[:, 0]
    
    # Calculate metrics
    mae = np.mean(np.abs(true_original - pred_original))
    rmse = np.sqrt(np.mean((true_original - pred_original) ** 2))
    r2 = 1 - np.sum((true_original - pred_original) ** 2) / np.sum((true_original - np.mean(true_original)) ** 2)
    
    # Print metrics
    print("\nModel Performance Metrics (Original Scale):")
    print(f"MAE: {mae:.2f} taxi requests")
    print(f"RMSE: {rmse:.2f} taxi requests")
    print(f"R²: {r2:.4f}")
    
    # Log to ClearML if available
    if task:
        task.get_logger().report_scalar(
            title="Original Scale Metrics",
            series="MAE",
            value=mae,
            iteration=0
        )
        task.get_logger().report_scalar(
            title="Original Scale Metrics",
            series="RMSE",
            value=rmse,
            iteration=0
        )
        task.get_logger().report_scalar(
            title="Original Scale Metrics",
            series="R2",
            value=r2,
            iteration=0
        )
    
    return mae, rmse, r2

def train_and_save_model(
    data_path: str = 'data_from_2024/taxi_demand_dataset.csv',
    model_path: str = 'models/transformer_model.pt',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    input_seq_len: int = 24,
    output_seq_len: int = 24,
    min_records_per_zone: int = 100,
    enable_clearml: bool = True
) -> None:
    print("Training transformer model...")
    print("="*50)
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    """Train and save the transformer model."""
    # Initialize ClearML task if enabled
    task = None
    if enable_clearml:
        try:
            from clearml import Task
            task = Task.init(
                project_name='TaxiDemandPrediction',
                task_name=f'ZoneLevelTransformer_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                task_type='training'
            )
        except Exception as e:
            print(f"\nWarning: ClearML initialization failed: {str(e)}")
            print("Continuing without ClearML logging...")
            enable_clearml = False
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    # sample rows, but not more than available
    # sample_size = min(500000, len(df))
    # df = df.sample(sample_size)
    # use only first 50 zones
    # df = df[df['zone_id'] < 50]
    df['hour'] = pd.to_datetime(df['hour'])
    
    print("\nInitial data shape:", df.shape)
    print("Time range:", df['hour'].min(), "to", df['hour'].max())
    print("Number of zones:", df['zone_id'].nunique())
    
    # Analyze zone-level data
    zone_counts = df.groupby('zone_id').size()
    print("\nZone-level statistics:")
    print(f"Records per zone (min): {zone_counts.min()}")
    print(f"Records per zone (max): {zone_counts.max()}")
    print(f"Records per zone (mean): {zone_counts.mean():.2f}")
    print(f"Records per zone (median): {zone_counts.median()}")
    
    # Filter zones with sufficient data
    valid_zones = zone_counts[zone_counts >= min_records_per_zone].index
    print(f"\nFound {len(valid_zones)} zones with {min_records_per_zone}+ records")
    
    if len(valid_zones) == 0:
        raise ValueError(f"No zones have {min_records_per_zone}+ records. Try reducing min_records_per_zone.")
    
    # Filter dataframe to include only valid zones
    df = df[df['zone_id'].isin(valid_zones)].copy()
    print(f"Filtered data shape: {df.shape}")
    
    # Define feature groups
    feature_columns = [
        # Weather Features
        'temp',             # Temperature
        'feels_like',       # Feels like temperature
        'wind_speed',       # Wind speed
        'rain_1h',          # Rainfall
        
        # Time Features
        'hour_of_day',      # Hour of day (0-23)
        'day_of_week',      # Day of week (0-6)
        'is_weekend',       # Weekend flag (0/1)
        'is_holiday',       # Holiday flag (0/1)
        'is_rush_hour'      # Rush hour flag (0/1)
    ]
    
    # Target column
    target_column = 'demand'  # Raw demand is our target
    
    # Static features
    static_columns = ['zone_id']  # Zone identifier
    
    # Check for NaN values
    nan_cols = df[feature_columns + static_columns].isna().sum()
    if nan_cols.any():
        print("\nWARNING: NaN values found in columns:")
        print(nan_cols[nan_cols > 0])
        print("Filling NaN values with appropriate methods...")
        
        # Fill NaN values appropriately
        for col in feature_columns:
            if df[col].isna().any():
                if 'demand' in col:
                    df[col] = df.groupby('zone_id')[col].transform(lambda x: x.fillna(x.mean()))
                else:
                    df[col] = df[col].fillna(df[col].mean())
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    # Sort by time and zone for proper splitting
    df = df.sort_values(['hour', 'zone_id'])
    
    # Split data by time while maintaining zone integrity
    train_cutoff = df['hour'].quantile(0.8)
    val_cutoff = df['hour'].quantile(0.9)
    
    train_data = df[df['hour'] <= train_cutoff]
    val_data = df[(df['hour'] > train_cutoff) & (df['hour'] <= val_cutoff)]
    test_data = df[df['hour'] > val_cutoff]
    
    print("\nData split sizes:")
    print(f"Train: {len(train_data)} records, {train_data['zone_id'].nunique()} zones")
    print(f"Val: {len(val_data)} records, {val_data['zone_id'].nunique()} zones")
    print(f"Test: {len(test_data)} records, {test_data['zone_id'].nunique()} zones")
    
    # Create datasets with reduced sequence lengths
    print("\nCreating datasets...")
    train_dataset = TimeSeriesDataset(
        train_data, feature_columns, static_columns,
        input_seq_len=input_seq_len, 
        output_seq_len=output_seq_len
    )
    val_dataset = TimeSeriesDataset(
        val_data, feature_columns, static_columns,
        input_seq_len=input_seq_len, 
        output_seq_len=output_seq_len
    )
    test_dataset = TimeSeriesDataset(
        test_data, feature_columns, static_columns,
        input_seq_len=input_seq_len, 
        output_seq_len=output_seq_len
    )
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model with reduced sequence lengths
    model = SimpleTimeSeriesTransformer(
        n_features=len(feature_columns),
        n_static_features=len(static_columns),
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len
    ).to(device)
    
    # Train model
    print("\nTraining model...")
    print(f"Input sequence length: {input_seq_len} hours")
    print(f"Output sequence length: {output_seq_len} hours")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Training on {len(valid_zones)} zones")
    
    history = train_model(model, train_loader, val_loader, device)
    print("Model training completed!")

    # After training, evaluate on test set
    print("\nEvaluating model on test set...")
    model.eval()
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for batch_x, batch_tgt_features, batch_y, batch_static in test_loader:
            batch_x = batch_x.to(device)
            batch_tgt_features = batch_tgt_features.to(device)
            batch_y = batch_y.to(device)
            batch_static = batch_static.to(device)
            
            y_pred = model(batch_x, batch_tgt_features, batch_static)
            all_true.append(batch_y.cpu().numpy())
            all_pred.append(y_pred.cpu().numpy())
    
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    
    # Calculate metrics in original scale
    calculate_original_scale_metrics(y_true, y_pred, scaler, task)
    
    # Create evaluation plots
    create_evaluation_plots(model, y_true, y_pred, history, task)
    
    # Save model and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_columns': feature_columns,
        'static_columns': static_columns,
        'history': history,
        'input_seq_len': input_seq_len,
        'output_seq_len': output_seq_len,
        'valid_zones': list(valid_zones)
    }, model_path)
    
    print(f"\nModel saved to {model_path}")
    
    # Close ClearML task if it was initialized
    if task:
        task.close()

def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def objective(trial: Trial, config: dict, device: str) -> float:
    logger.info(f"\nStarting Trial {trial.number}")
    logger.info(f"Memory usage at start: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Get hyperparameters from config
    model_config = config['model']
    training_config = config['training']
    
    # Ensure device is valid
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Convert learning rate bounds to float
    lr_min = float(training_config['learning_rate']['min'])
    lr_max = float(training_config['learning_rate']['max'])
    
    # Restrict d_model and n_heads to smaller values for now
    n_heads = trial.suggest_int('n_heads', model_config['n_heads']['min'], model_config['n_heads']['max'])
    d_model = trial.suggest_int('d_model', model_config['d_model']['min'], model_config['d_model']['max'])
    d_model = (d_model // n_heads) * n_heads
    
    # Ensure input_seq_len >= output_seq_len to avoid dimension mismatch
    input_seq_len = trial.suggest_categorical('input_seq_len', model_config['input_seq_len'])
    output_seq_len = trial.suggest_categorical('output_seq_len', model_config['output_seq_len'])
    if input_seq_len < output_seq_len:
        input_seq_len = output_seq_len  # Ensure input is at least as long as output
    
    params = {
        'd_model': d_model,
        'n_heads': n_heads,
        'n_encoder_layers': trial.suggest_int('n_encoder_layers', 
                                            model_config['n_encoder_layers']['min'], 
                                            model_config['n_encoder_layers']['max']),
        'n_decoder_layers': trial.suggest_int('n_decoder_layers', 
                                            model_config['n_decoder_layers']['min'], 
                                            model_config['n_decoder_layers']['max']),
        'dropout': trial.suggest_float('dropout', 
                                     model_config['dropout']['min'], 
                                     model_config['dropout']['max']),
        'learning_rate': trial.suggest_float('learning_rate', 
                                           lr_min,
                                           lr_max, 
                                           log=training_config['learning_rate']['log']),
        'batch_size': trial.suggest_categorical('batch_size', training_config['batch_size']),
        'input_seq_len': input_seq_len,
        'output_seq_len': output_seq_len,
    }
    
    logger.info("Selected hyperparameters for this trial:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize ClearML task if enabled
    task = None
    if config['optimization']['enable_clearml']:
        try:
            task = Task.init(
                project_name='TaxiDemandPrediction',
                task_name=f'OptunaTrial_{trial.number}',
                task_type='training'
            )
            task.connect(params)
            logger.info("ClearML task initialized successfully")
        except Exception as e:
            logger.warning(f"ClearML initialization failed: {str(e)}")
            logger.info("Continuing without ClearML logging...")
            config['optimization']['enable_clearml'] = False
    
    try:
        logger.info("Loading data...")
        # Load and preprocess data
        df = pd.read_csv(config['data']['path'])
        logger.info(f"Data loaded. Shape: {df.shape}")
        
        logger.info("Preprocessing data...")
        # Define feature groups from config
        feature_columns = config['features']['dynamic']
        target_column = config['features']['target']
        static_columns = config['features']['static']
        
        # Create datasets
        logger.info("Creating datasets...")
        train_dataset = TimeSeriesDataset(
            df, feature_columns, static_columns,
            input_seq_len=input_seq_len, 
            output_seq_len=output_seq_len
        )
        logger.info(f"Created dataset with {len(train_dataset)} sequences")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(train_dataset, batch_size=params['batch_size'])
        
        # Initialize model
        logger.info("Initializing model...")
        model = SimpleTimeSeriesTransformer(
            d_model=params['d_model'],
            n_heads=params['n_heads'],
            n_encoder_layers=params['n_encoder_layers'],
            n_decoder_layers=params['n_decoder_layers'],
            dropout=params['dropout'],
            input_seq_len=params['input_seq_len'],
            output_seq_len=params['output_seq_len'],
            n_features=len(feature_columns),
            n_static_features=len(static_columns)
        ).to(device)
        
        # Train model
        logger.info("Starting model training...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            n_epochs=training_config['max_epochs'],
            learning_rate=params['learning_rate'],
            patience=training_config['patience']
        )
        
        # Log final metrics
        best_val_loss = min(history['val_loss'])
        logger.info(f"Trial {trial.number} completed. Best validation loss: {best_val_loss:.4f}")
        
        return best_val_loss
        
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        logger.error(f"Error occurred at: {e.__traceback__.tb_lineno}")
        raise optuna.TrialPruned()
    finally:
        if task and config['optimization']['enable_clearml']:
            try:
                logger.info(f"Closing ClearML task for trial {trial.number}")
                task.close()
            except Exception as e:
                logger.warning(f"Failed to close ClearML task: {str(e)}")
        logger.info(f"Trial {trial.number} cleanup complete. Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def optimize_hyperparameters(config: dict) -> None:
    """Run hyperparameter optimization using Optuna."""
    logger.info(f"Starting hyperparameter optimization with {config['optimization']['n_trials']} trials...")
    logger.info(f"Each trial will run for up to {config['training']['max_epochs']} epochs with early stopping (patience={config['training']['patience']})")
    logger.info(f"Timeout per trial: {config['optimization']['timeout_per_trial']} seconds")
    
    # Create study with better error handling
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=config['optimization']['n_startup_trials'],
            n_warmup_steps=config['optimization']['n_warmup_steps'],
            interval_steps=1
        )
    )
    
    # Track completed trials
    completed_trials = 0
    
    def objective_with_progress(trial):
        nonlocal completed_trials
        logger.info(f"\n========== Starting Trial {trial.number} ==========")
        try:
            # Add timeout to the objective function
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Trial {trial.number} timed out after {config['optimization']['timeout_per_trial']} seconds")
            
            # Set the timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(config['optimization']['timeout_per_trial'])
            
            try:
                result = objective(trial, config, config['training']['device'])
                completed_trials += 1
                logger.info(f"========== Trial {trial.number} Complete ==========")
                logger.info(f"Progress: {completed_trials}/{config['optimization']['n_trials']} trials completed\n")
                return result
            finally:
                # Disable the alarm
                signal.alarm(0)
                
        except TimeoutError as e:
            logger.warning(f"\nTrial {trial.number} timed out: {str(e)}")
            completed_trials += 1
            logger.info(f"========== Trial {trial.number} Timed Out ==========")
            logger.info(f"Progress: {completed_trials}/{config['optimization']['n_trials']} trials completed\n")
            raise optuna.TrialPruned()
        except Exception as e:
            logger.error(f"\nTrial {trial.number} failed with error: {str(e)}")
            completed_trials += 1
            logger.info(f"========== Trial {trial.number} Failed ==========")
            logger.info(f"Progress: {completed_trials}/{config['optimization']['n_trials']} trials completed\n")
            raise optuna.TrialPruned()
    
    logger.info("\n===== Starting Optuna Optimization Loop =====")
    try:
        study.optimize(
            objective_with_progress,
            n_trials=config['optimization']['n_trials'],
            show_progress_bar=True,
            gc_after_trial=True,
            timeout=config['optimization']['timeout_per_trial'] * config['optimization']['n_trials']
        )
    except Exception as e:
        logger.error(f"\nOptimization stopped due to error: {str(e)}")
        logger.info(f"Completed {completed_trials} trials before stopping")
    logger.info("===== Optuna Optimization Loop Finished =====\n")
    
    # Print results
    logger.info("\nOptimization completed!")
    logger.info(f"Successfully completed {completed_trials} out of {config['optimization']['n_trials']} trials")
    
    if completed_trials > 0:
        logger.info("\nBest trial:")
        trial = study.best_trial
        logger.info(f"  Value: {trial.value:.4f}")
        logger.info("\nBest hyperparameters:")
        for key, value in trial.params.items():
            logger.info(f"  {key}: {value}")
        
        # Create optimization plots
        if config['optimization']['enable_clearml']:
            try:
                task = Task.init(
                    project_name='TaxiDemandPrediction',
                    task_name='HyperparameterOptimizationResults',
                    task_type='training'
                )
                
                # Plot optimization history
                fig = vis.plot_optimization_history(study)
                task.get_logger().report_plotly(
                    title="Optimization History",
                    series="Optimization",
                    figure=fig
                )
                
                # Plot parameter importance
                fig = vis.plot_param_importances(study)
                task.get_logger().report_plotly(
                    title="Parameter Importance",
                    series="Optimization",
                    figure=fig
                )
                
                task.close()
            except Exception as e:
                logger.warning(f"\nFailed to log optimization plots: {str(e)}")
                logger.info("Continuing without ClearML logging...")
        
        return study.best_params
    else:
        logger.warning("\nNo successful trials completed!")
        return None

if __name__ == "__main__":
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        
        # Run hyperparameter optimization
        logger.info("\nRunning hyperparameter optimization...")
        best_params = optimize_hyperparameters(config)
        
        if best_params:
            logger.info("\nTraining final model with best parameters...")
            train_and_save_model(
                data_path=config['data']['path'],
                enable_clearml=config['optimization']['enable_clearml'],
                input_seq_len=best_params['input_seq_len'],
                output_seq_len=best_params['output_seq_len']
            )
        logger.info("All trials complete. Exiting script.")
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise 