import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from clearml import Task, Dataset, OutputModel
import joblib
import os
from datetime import datetime, timedelta
import glob
from sklearn.preprocessing import StandardScaler
import math
from tqdm import tqdm

class TimeSeriesTransformer(nn.Module):
    """
    Transformer model adapted for time series forecasting.
    Uses relative positional encoding and a simplified architecture suitable for time series.
    """
    def __init__(
        self,
        d_model: int = 16,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        dropout: float = 0.1,
        input_seq_len: int = 168,  # 1 week of hourly data
        output_seq_len: int = 24,  # Predict next 24 hours
    ):
        super().__init__()
        
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        
        # Input projection layer
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=input_seq_len + output_seq_len)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # src shape: [batch_size, input_seq_len, 1]
        # tgt shape: [batch_size, output_seq_len, 1]
        
        # Project input and target to d_model dimensions
        src = self.input_projection(src)  # [batch_size, input_seq_len, d_model]
        tgt = self.input_projection(tgt)  # [batch_size, output_seq_len, d_model]
        
        # Add positional encoding
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        
        # Create masks
        tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer forward pass
        memory = self.encoder(src)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # Project back to original dimension
        output = self.output_projection(output)  # [batch_size, output_seq_len, 1]
        
        return output

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncoding(nn.Module):
    """
    Relative positional encoding for time series data.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class TimeSeriesDataset(Dataset):
    """
    Dataset class for time series data.
    Creates sequences of input_seq_len timesteps and corresponding output_seq_len targets.
    """
    def __init__(
        self,
        data: np.ndarray,
        input_seq_len: int = 168,  # 1 week
        output_seq_len: int = 24,  # 24 hours
        stride: int = 1
    ):
        # Ensure data is 1D
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
        
        # Reshape to [seq_len, 1] for both input and target
        input_seq = sequence[:self.input_seq_len].view(-1, 1)
        target_seq = sequence[self.input_seq_len:].view(-1, 1)
        
        return input_seq, target_seq

def load_and_preprocess_data(data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess all parquet files.
    Returns train and test DataFrames with hourly demand.
    """
    # Get all parquet files
    parquet_files = sorted(glob.glob(os.path.join(data_dir, 'yellow_tripdata_*.parquet')))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load and process all files
    all_dfs = []
    for file in tqdm(parquet_files, desc="Loading data files"):
        df = pd.read_parquet(file)
        
        # Ensure datetime is parsed correctly
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        
        # Convert to hourly demand
        df['hour'] = df['tpep_pickup_datetime'].dt.floor('H')
        demand = df.groupby('hour').size().reset_index()
        demand.columns = ['timestamp', 'demand']
        all_dfs.append(demand)
    
    # Combine all dataframes and aggregate any duplicate hours
    print("Merging dataframes...")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Handle duplicate timestamps by summing the demand
    merged_df = merged_df.groupby('timestamp')['demand'].sum().reset_index()
    merged_df = merged_df.sort_values('timestamp')
    
    # Verify date range
    print(f"\nRaw date range:")
    print(f"Earliest date: {merged_df['timestamp'].min()}")
    print(f"Latest date: {merged_df['timestamp'].max()}\n")
    
    # Ensure continuous hourly timestamps
    full_range = pd.date_range(
        start=merged_df['timestamp'].min(),
        end=merged_df['timestamp'].max(),
        freq='H'
    )
    
    # Create a DataFrame with the full range and merge with existing data
    date_df = pd.DataFrame({'timestamp': full_range})
    merged_df = pd.merge(date_df, merged_df, on='timestamp', how='left')
    merged_df['demand'] = merged_df['demand'].fillna(0)
    
    # Split into train and test (last month is test)
    split_date = pd.Timestamp('2025-01-01')
    train_df = merged_df[merged_df['timestamp'] < split_date]
    test_df = merged_df[merged_df['timestamp'] >= split_date]
    
    print(f"Train set: {len(train_df):,} hours from {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Test set: {len(test_df):,} hours from {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    return train_df, test_df

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 5,
    learning_rate: float = 0.001,
    patience: int = 5
) -> Dict[str, List[float]]:
    """
    Train the model and return training history.
    Implements early stopping based on validation loss.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Train]')
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(batch_x, batch_y)
            loss = criterion(y_pred, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_losses = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Val]')
        with torch.no_grad():
            for batch_x, batch_y in val_pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                y_pred = model(batch_x, batch_y)
                val_loss = criterion(y_pred, batch_y)
                val_losses.append(val_loss.item())
                val_pbar.set_postfix({'loss': f'{val_loss.item():.4f}'})
        
        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print("New best validation loss!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return history

def train_and_save_model(
    data_dir: str = 'data',
    model_path: str = 'transformer_model.pt',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    """
    Train the Transformer model and save it to disk.
    Also logs experiment to ClearML.
    """
    # Create unique task name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = f'TransformerTraining_{timestamp}'
    
    # Initialize ClearML task
    task = Task.init(
        project_name='TaxiDemandPrediction',
        task_name=task_name,
        task_type='training'
    )
    
    # Load and preprocess data
    train_df, test_df = load_and_preprocess_data(data_dir)
    
    # Scale the data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_df['demand'].values.reshape(-1, 1))
    test_data = scaler.transform(test_df['demand'].values.reshape(-1, 1))
    
    # Create datasets
    input_seq_len = 168  # 1 week
    output_seq_len = 24  # 24 hours
    
    train_size = int(0.8 * len(train_data))
    train_dataset = TimeSeriesDataset(train_data[:train_size], input_seq_len, output_seq_len)
    val_dataset = TimeSeriesDataset(train_data[train_size:], input_seq_len, output_seq_len)
    test_dataset = TimeSeriesDataset(test_data, input_seq_len, output_seq_len)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = TimeSeriesTransformer().to(device)
    
    # Log model parameters
    task.connect({
        "model_params": {
            "d_model": 64,
            "n_heads": 8,
            "n_encoder_layers": 3,
            "n_decoder_layers": 3,
            "dropout": 0.1,
            "input_seq_len": input_seq_len,
            "output_seq_len": output_seq_len
        },
        "training_params": {
            "batch_size": batch_size,
            "learning_rate": 0.01,
            "n_epochs": 5,
            "patience": 5
        }
    })
    
    # Train model
    print(f"Training model (Task ID: {task.id})...")
    history = train_model(model, train_loader, val_loader, device)
    
    # Log training history
    for epoch, (train_loss, val_loss) in enumerate(zip(history['train_loss'], history['val_loss'])):
        logger = task.get_logger()
        logger.report_scalar("Loss", "train", iteration=epoch, value=train_loss)
        logger.report_scalar("Loss", "validation", iteration=epoch, value=val_loss)
    
    # Evaluate on test set
    model.eval()
    criterion = nn.MSELoss()
    test_losses = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            y_pred = model(batch_x, batch_y)
            test_loss = criterion(y_pred, batch_y)
            test_losses.append(test_loss.item())
    
    avg_test_loss = np.mean(test_losses)
    rmse = np.sqrt(avg_test_loss) * scaler.scale_[0]  # Convert back to original scale
    
    print(f"Test RMSE: {rmse:.2f}")
    
    # Log metrics
    logger = task.get_logger()
    logger.report_scalar("Metrics", "Test RMSE", iteration=0, value=rmse)
    
    # Save model and scaler
    model_components = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'model_params': {
            'd_model': 64,
            'n_heads': 8,
            'n_encoder_layers': 3,
            'n_decoder_layers': 3,
            'dropout': 0.1,
            'input_seq_len': input_seq_len,
            'output_seq_len': output_seq_len
        },
        'task_id': task.id,  # Store task ID with model
        'training_timestamp': timestamp
    }
    
    torch.save(model_components, model_path)
    print(f"Model saved to {model_path}")
    
    # Register model in ClearML
    output_model = OutputModel(
        task=task,
        name=f'TaxiDemandTransformer_{timestamp}',
        framework='pytorch',
        label_enumeration={}
    )
    output_model.update_weights(weights_filename=model_path)
    
    print(f"Model registered in ClearML (Task ID: {task.id})")
    task.close()

def load_model(
    model_path: str = 'transformer_model.pt',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[TimeSeriesTransformer, StandardScaler]:
    """
    Load the trained Transformer model and scaler from disk.
    """
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
    
    # Load model components
    model_components = torch.load(model_path, map_location=device)
    
    # Initialize model with saved parameters
    model = TimeSeriesTransformer(**model_components['model_params']).to(device)
    model.load_state_dict(model_components['model_state_dict'])
    model.eval()
    
    return model, model_components['scaler']

def predict_demand(
    input_features: Dict[str, any],
    model_path: str = 'transformer_model.pt',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, List[float]]:
    """
    Predict taxi demand for the next 24 hours.
    
    Args:
        input_features: Dictionary containing historical demand data
        model_path: Path to the saved model
        device: Device to run predictions on
        
    Returns:
        Dictionary containing predicted demand
    """
    model, scaler = load_model(model_path, device)
    
    # Prepare input data
    if 'historical_demand' not in input_features:
        raise ValueError("Missing required feature: historical_demand (needs 1 week of hourly data)")
    
    historical_demand = np.array(input_features['historical_demand'])
    if len(historical_demand) != model.input_seq_len:
        raise ValueError(f"Historical demand must contain {model.input_seq_len} hours of data")
    
    # Scale input data
    scaled_input = scaler.transform(historical_demand.reshape(-1, 1))
    
    # Prepare input tensor
    x = torch.FloatTensor(scaled_input).unsqueeze(0).to(device)  # Add batch dimension
    
    # Initialize target with zeros
    tgt = torch.zeros((1, model.output_seq_len, 1), device=device)
    
    # Generate prediction
    with torch.no_grad():
        scaled_pred = model(x, tgt)
    
    # Convert predictions back to original scale
    predictions = scaler.inverse_transform(scaled_pred.cpu().squeeze().numpy())
    
    # Generate timestamps for predictions
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
    # Generate some dummy historical data (1 week of hourly data)
    historical_demand = np.random.normal(100, 20, 168)  # 168 hours = 1 week
    
    test_features = {
        'timestamp': '2025-02-01 00:00:00',
        'historical_demand': historical_demand.tolist()
    }
    
    prediction = predict_demand(test_features)
    print("\nTest prediction for next 24 hours:")
    for i in range(len(prediction['timestamp'])):
        print(f"Time: {prediction['timestamp'][i]}, Demand: {prediction['demand'][i]:.2f}") 