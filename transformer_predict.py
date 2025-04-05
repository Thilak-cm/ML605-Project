import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from clearml import Task, OutputModel
import joblib
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class SimpleTimeSeriesTransformer(nn.Module):
    """
    Simplified Transformer model for time series forecasting.
    Uses minimal layers and parameters for quick training and iteration.
    """
    def __init__(
        self,
        d_model: int = 8,  # Reduced from 16
        n_heads: int = 2,  # Reduced from 4
        n_encoder_layers: int = 1,  # Reduced from 3
        n_decoder_layers: int = 1,  # Reduced from 3
        dropout: float = 0.1,
        input_seq_len: int = 24,  # Reduced from 168 to 24 hours
        output_seq_len: int = 24,
    ):
        super().__init__()
        
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        
        # Simple input projection
        self.input_projection = nn.Linear(1, d_model)
        
        # Simplified positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, input_seq_len + output_seq_len, d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,  # Reduced from 4
            dropout=dropout,
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,  # Reduced from 4
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # Project input and target to d_model dimensions
        src = self.input_projection(src)
        tgt = self.input_projection(tgt)
        
        # Add positional encoding
        src = src + self.pos_encoder[:, :src.size(1)]
        tgt = tgt + self.pos_encoder[:, :tgt.size(1)]
        
        # Create mask for decoder
        tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer forward pass
        memory = self.encoder(src)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # Project back to original dimension
        output = self.output_projection(output)
        
        return output

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class TimeSeriesDataset(Dataset):
    """Dataset class for time series data."""
    def __init__(
        self,
        data: np.ndarray,
        input_seq_len: int = 24,  # Reduced from 168
        output_seq_len: int = 24,
        stride: int = 24  # Increased stride for less overlap
    ):
        self.data = torch.FloatTensor(data.reshape(-1))
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.stride = stride
        
    def __len__(self) -> int:
        return (len(self.data) - self.input_seq_len - self.output_seq_len) // self.stride + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.stride
        end_idx = start_idx + self.input_seq_len + self.output_seq_len
        sequence = self.data[start_idx:end_idx]
        return (
            sequence[:self.input_seq_len].view(-1, 1),
            sequence[self.input_seq_len:].view(-1, 1)
        )

def create_evaluation_plots(model, y_true, y_pred, history, task, dates=None):
    """
    Create comprehensive evaluation plots and log them to ClearML
    
    Args:
        model: Trained transformer model
        y_true: True values
        y_pred: Predicted values
        history: Training history dictionary
        task: ClearML task object
        dates: Optional datetime index for time-based plots
    """
    logger = task.get_logger()
    
    # 1. Training History Plot
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    logger.report_matplotlib_figure(
        title="Training History",
        series="Training",
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    # 2. Prediction vs Actual Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual')
    logger.report_matplotlib_figure(
        title="Prediction vs Actual",
        series="Evaluation",
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    # 3. Residual Plot
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    logger.report_matplotlib_figure(
        title="Residuals",
        series="Evaluation",
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    # 4. Residual Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residual Value')
    plt.ylabel('Count')
    plt.title('Residual Distribution')
    logger.report_matplotlib_figure(
        title="Residual Distribution",
        series="Evaluation",
        figure=plt.gcf(),
        iteration=0
    )
    plt.close()
    
    # 5. Error Metrics Over Time
    if dates is not None:
        # Flatten and align dates with residuals
        flat_dates = []
        for date_series in dates:
            flat_dates.extend(date_series)
        flat_dates = flat_dates[:len(residuals)]  # Ensure same length as residuals
        
        plt.figure(figsize=(15, 6))
        plt.plot(flat_dates, np.abs(residuals), alpha=0.5)
        plt.xlabel('Date')
        plt.ylabel('Absolute Error')
        plt.title('Absolute Error Over Time')
        plt.xticks(rotation=45)
        logger.report_matplotlib_figure(
            title="Error Over Time",
            series="Time Analysis",
            figure=plt.gcf(),
            iteration=0
        )
        plt.close()
        
        # 6. Hourly Error Patterns
        hourly_errors = pd.DataFrame({
            'hour': pd.DatetimeIndex(flat_dates).hour,  # Use flattened dates
            'abs_error': np.abs(residuals)
        })
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='hour', y='abs_error', data=hourly_errors)
        plt.xlabel('Hour of Day')
        plt.ylabel('Absolute Error')
        plt.title('Error Distribution by Hour')
        logger.report_matplotlib_figure(
            title="Hourly Error Patterns",
            series="Time Analysis",
            figure=plt.gcf(),
            iteration=0
        )
        plt.close()
    
    # 7. Prediction Intervals (if available)
    if hasattr(model, 'predict_interval'):
        try:
            lower, upper = model.predict_interval(y_pred)
            plt.figure(figsize=(15, 6))
            plt.fill_between(range(len(y_pred)), lower, upper, alpha=0.3, label='95% CI')
            plt.plot(y_true, label='Actual', alpha=0.7)
            plt.plot(y_pred, label='Predicted', alpha=0.7)
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.title('Predictions with Confidence Intervals')
            plt.legend()
            logger.report_matplotlib_figure(
                title="Prediction Intervals",
                series="Evaluation",
                figure=plt.gcf(),
                iteration=0
            )
            plt.close()
        except:
            pass

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 30,  # Increased from 10 to 30
    learning_rate: float = 0.001,
    patience: int = 5  # Increased from 3 to 5
) -> Dict[str, List[float]]:
    """Simplified training loop with early stopping."""
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
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            y_pred = model(batch_x, batch_y)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                y_pred = model(batch_x, batch_y)
                val_losses.append(criterion(y_pred, batch_y).item())
        
        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs}")
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

def train_and_save_model(
    data_path: str = 'data_from_2024/merged_features.csv',
    model_path: str = 'transformer_model.pt',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    """Train and save the simplified transformer model."""
    # Initialize ClearML task
    task = Task.init(
        project_name='TaxiDemandPrediction',
        task_name=f'SimpleTransformerTraining_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        task_type='training'
    )
    
    # Load and preprocess data
    print("Loading data...")
    df = pd.read_csv(data_path)
    df['hour'] = pd.to_datetime(df['hour'])
    demand = df['demand'].values
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(demand.reshape(-1, 1))
    
    # Create datasets with simplified sequence lengths
    input_seq_len = 24  # 1 day
    output_seq_len = 24  # 1 day prediction
    
    # Split data
    train_size = int(0.8 * len(scaled_data))
    val_size = int(0.1 * len(scaled_data))
    
    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:train_size + val_size]
    test_data = scaled_data[train_size + val_size:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, input_seq_len, output_seq_len)
    val_dataset = TimeSeriesDataset(val_data, input_seq_len, output_seq_len)
    test_dataset = TimeSeriesDataset(test_data, input_seq_len, output_seq_len)
    
    # Create data loaders with larger batch size
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = SimpleTimeSeriesTransformer(
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len
    ).to(device)
    
    # Log parameters
    task.connect({
        "model_params": {
            "d_model": 8,
            "n_heads": 2,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "dropout": 0.1,
            "input_seq_len": input_seq_len,
            "output_seq_len": output_seq_len
        },
        "training_params": {
            "batch_size": batch_size,
            "learning_rate": 0.001,
            "n_epochs": 30,
            "patience": 5
        }
    })
    
    # Train model
    print("Training model...")
    history = train_model(model, train_loader, val_loader, device)
    
    # Evaluate on test set
    model.eval()
    all_y_true = []
    all_y_pred = []
    test_dates = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            y_pred = model(batch_x, batch_y)
            all_y_true.extend(scaler.inverse_transform(batch_y.cpu().numpy().reshape(-1, 1)).flatten())
            all_y_pred.extend(scaler.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).flatten())
            # Generate dates for this batch
            test_dates.extend([df['hour'].iloc[-len(test_data):] + pd.Timedelta(hours=i) for i in range(len(batch_y))])
    
    # Create evaluation plots
    create_evaluation_plots(
        model,
        np.array(all_y_true),
        np.array(all_y_pred),
        history,
        task,
        dates=test_dates
    )
    
    # Log metrics
    logger = task.get_logger()
    for epoch, (train_loss, val_loss) in enumerate(zip(history['train_loss'], history['val_loss'])):
        logger.report_scalar("Loss", "train", iteration=epoch, value=train_loss)
        logger.report_scalar("Loss", "validation", iteration=epoch, value=val_loss)
    
    # Calculate and log final metrics
    mse = mean_squared_error(all_y_true, all_y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_y_true, all_y_pred)
    r2 = r2_score(all_y_true, all_y_pred)
    
    logger.report_scalar("Final Metrics", "RMSE", iteration=0, value=rmse)
    logger.report_scalar("Final Metrics", "MAE", iteration=0, value=mae)
    logger.report_scalar("Final Metrics", "R2", iteration=0, value=r2)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'input_seq_len': input_seq_len,
        'output_seq_len': output_seq_len,
        'history': history,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    }, model_path)
    
    # Register model in ClearML
    output_model = OutputModel(
        task=task,
        name='SimpleTaxiDemandTransformer',
        framework='pytorch'
    )
    output_model.update_weights(weights_filename=model_path)
    
    print(f"\nTraining completed with final metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")
    
    task.close()

def predict_demand(
    input_features: Dict[str, any],
    model_path: str = 'transformer_model.pt',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, List[float]]:
    """Make predictions using the simplified transformer model."""
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = SimpleTimeSeriesTransformer(
        input_seq_len=checkpoint['input_seq_len'],
        output_seq_len=checkpoint['output_seq_len']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    
    # Prepare input data
    if 'historical_demand' not in input_features:
        raise ValueError("Missing required feature: historical_demand")
    
    historical_demand = np.array(input_features['historical_demand'])
    if len(historical_demand) != model.input_seq_len:
        raise ValueError(f"Historical demand must contain {model.input_seq_len} hours of data")
    
    # Scale input
    scaled_input = scaler.transform(historical_demand.reshape(-1, 1))
    x = torch.FloatTensor(scaled_input).unsqueeze(0).to(device)
    
    # Initialize target with zeros
    tgt = torch.zeros((1, model.output_seq_len, 1), device=device)
    
    # Generate prediction
    with torch.no_grad():
        scaled_pred = model(x, tgt)
    
    # Ensure correct shape for inverse_transform
    predictions = scaler.inverse_transform(scaled_pred.cpu().squeeze().numpy().reshape(-1, 1))
    predictions = predictions.flatten()  # Convert back to 1D array for output
    
    # Generate timestamps
    start_time = pd.Timestamp(input_features.get('timestamp', pd.Timestamp.now()))
    timestamps = [start_time + timedelta(hours=i) for i in range(len(predictions))]
    
    return {
        'timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
        'demand': predictions.tolist()
    }

if __name__ == "__main__":
    # Train model
    train_and_save_model()
    
    # Test prediction
    historical_demand = np.random.normal(100, 20, 24)  # 24 hours of historical data
    
    test_features = {
        'timestamp': '2025-02-01 00:00:00',
        'historical_demand': historical_demand.tolist()
    }
    
    prediction = predict_demand(test_features)
    print("\nTest prediction for next 24 hours:")
    for ts, demand in zip(prediction['timestamp'], prediction['demand']):
        print(f"{ts}: {demand:.2f}") 