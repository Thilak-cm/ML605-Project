import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Union
from functools import lru_cache
import logging
import os
from datetime import datetime, timedelta
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = 'transformer_model.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleTimeSeriesTransformer(nn.Module):
    """Transformer model for multi-feature time series forecasting."""
    def __init__(
        self,
        d_model: int = 8,
        n_heads: int = 2,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        dropout: float = 0.1,
        input_seq_len: int = 12,
        output_seq_len: int = 12,
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
        src = self.feature_projection(src)
        tgt = self.feature_projection(tgt_features)
        
        # Project and expand static features
        static = self.static_projection(static_features)
        static = static.unsqueeze(1).expand(-1, src.size(1), -1)
        
        # Add static features and positional encoding
        src = src + static + self.pos_encoder[:, :src.size(1)]
        tgt = tgt + static[:, :tgt.size(1)] + self.pos_encoder[:, :tgt.size(1)]
        
        # Create mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer forward pass
        memory = self.encoder(src)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # Project back to original dimension
        output = self.output_projection(output)
        
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

@lru_cache(maxsize=1)
def load_model() -> tuple:
    """Load the trained transformer model and metadata with caching."""
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        model = SimpleTimeSeriesTransformer(
            n_features=len(checkpoint['feature_columns']),
            n_static_features=len(checkpoint['static_columns']),
            input_seq_len=checkpoint['input_seq_len'],
            output_seq_len=checkpoint['output_seq_len']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(DEVICE)
        
        logger.info(f"Successfully loaded model from {MODEL_PATH}")
        return model, checkpoint['scaler'], checkpoint['feature_columns'], checkpoint['static_columns']
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def predict_demand(
    zone_id: int,
    historical_features: Dict[str, List[float]],
    timestamp: str = None
) -> Dict[str, Union[List[str], List[float]]]:
    """
    Predict taxi demand for a specific zone using the loaded transformer model.
    
    Args:
        zone_id: int, ID of the zone to predict for
        historical_features: Dict with feature values for last N hours:
            {
                'demand': [...],  # Raw demand values
                'temp': [...],    # Temperature values
                'rain_1h': [...], # Rainfall values
                ...
            }
        timestamp: str, optional, prediction start time (default: current time)
            
    Returns:
        Dict with:
            - timestamp: List[str], hourly timestamps for predictions
            - demand: List[float], predicted demand values
            - confidence: List[Dict], confidence intervals (if available)
    """
    try:
        # Load model and metadata (cached)
        model, scaler, feature_columns, static_columns = load_model()
        
        # Validate input
        if not historical_features:
            raise ValueError("No historical features provided")
        
        # Check if we have all required features
        missing_features = [f for f in feature_columns if f not in historical_features]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Prepare input features
        input_data = np.array([historical_features[f] for f in feature_columns]).T
        if len(input_data) != model.input_seq_len:
            raise ValueError(f"Expected {model.input_seq_len} hours of historical data, got {len(input_data)}")
        
        # Scale features
        scaled_input = scaler.transform(input_data)
        x = torch.FloatTensor(scaled_input).unsqueeze(0).to(DEVICE)  # [1, seq_len, n_features]
        
        # Prepare static features (zone_id)
        static = torch.FloatTensor([[zone_id]]).to(DEVICE)
        
        # Initialize target features with zeros
        tgt_features = torch.zeros((1, model.output_seq_len, len(feature_columns))).to(DEVICE)
        
        # Generate prediction
        with torch.no_grad():
            scaled_pred = model(x, tgt_features, static)
        
        # Get predictions (we only want the demand column)
        predictions = scaled_pred.cpu().numpy().squeeze()
        
        # Generate timestamps
        start_time = pd.Timestamp(timestamp) if timestamp else pd.Timestamp.now()
        timestamps = [start_time + timedelta(hours=i) for i in range(len(predictions))]
        
        result = {
            'timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
            'demand': predictions.tolist(),
            'zone_id': zone_id
        }
        
        logger.info(f"Successfully generated predictions for zone {zone_id} ({len(predictions)} hours)")
        return result
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Load some sample data
    try:
        print("\n" + "="*50)
        print("Loading sample data and model...")
        print("="*50)
        
        df = pd.read_csv('data_from_2024/merged_features.csv')
        df['hour'] = pd.to_datetime(df['hour'])
        
        # Get a random sample zone
        sample_zone = df['zone_id'].sample(n=1).iloc[0]
        zone_data = df[df['zone_id'] == sample_zone].tail(12)  # Last 12 hours
        
        # Load model to get required features
        model, _, feature_columns, _ = load_model()
        
        # Prepare features
        features = {}
        print("\nPreparing input features for zone", sample_zone)
        print("-"*50)
        for col in feature_columns:
            if col in zone_data.columns:
                features[col] = zone_data[col].tolist()
                print(f"{col:15s}: {zone_data[col].values[:5]}...")  # Show first 5 values
            else:
                features[col] = [0] * len(zone_data)  # Default to zeros if feature not found
                print(f"{col:15s}: [0, 0, 0, 0, 0]... (default values)")
        
        # Print input summary
        print("\nInput Summary:")
        print("-"*50)
        print(f"Zone ID: {sample_zone}")
        print(f"Time Range: {zone_data['hour'].min()} to {zone_data['hour'].max()}")
        print(f"Number of Features: {len(feature_columns)}")
        print(f"Sequence Length: {len(zone_data)} hours")
        
        # Make prediction
        print("\n" + "="*50)
        print("Making predictions...")
        print("="*50)
        
        result = predict_demand(
            zone_id=sample_zone,
            historical_features=features,
            timestamp=zone_data['hour'].iloc[-1]
        )
        
        # Print results in a detailed format
        print("\nPrediction Results:")
        print("-"*50)
        print(f"Zone ID: {result['zone_id']}")
        print(f"Prediction Start Time: {result['timestamp'][0]}")
        print(f"Number of Hours Predicted: {len(result['demand'])}")
        
        print("\nHourly Predictions:")
        print("-"*50)
        print("Time                | Predicted Demand")
        print("-"*50)
        for ts, demand in zip(result['timestamp'], result['demand']):
            print(f"{ts:20s} | {demand:14.2f}")
        
        # Print statistics
        print("\nPrediction Statistics:")
        print("-"*50)
        print(f"Min Demand: {min(result['demand']):.2f}")
        print(f"Max Demand: {max(result['demand']):.2f}")
        print(f"Mean Demand: {np.mean(result['demand']):.2f}")
        print(f"Std Dev: {np.std(result['demand']):.2f}")
        
    except Exception as e:
        print(f"\nError in example: {str(e)}")
        import traceback
        traceback.print_exc() 