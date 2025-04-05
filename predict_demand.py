import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
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
    """Simplified Transformer model for time series forecasting."""
    def __init__(
        self,
        d_model: int = 8,
        n_heads: int = 2,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        dropout: float = 0.1,
        input_seq_len: int = 24,
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
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # Project input and target to d_model dimensions
        src = self.input_projection(src)
        tgt = self.input_projection(tgt)
        
        # Add positional encoding
        src = src + self.pos_encoder[:, :src.size(1)]
        tgt = tgt + self.pos_encoder[:, :tgt.size(1)]
        
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
    """Load the trained transformer model and scaler with caching."""
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        model = SimpleTimeSeriesTransformer(
            input_seq_len=checkpoint['input_seq_len'],
            output_seq_len=checkpoint['output_seq_len']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(DEVICE)
        
        logger.info(f"Successfully loaded model from {MODEL_PATH}")
        return model, checkpoint['scaler']
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def predict_demand(input_features: Dict) -> Dict[str, List]:
    """
    Predict taxi demand using the loaded transformer model.
    
    Args:
        input_features: Dictionary containing:
            - historical_demand: List[float], last 24 hours of demand values
            - timestamp: str, optional, prediction start time (default: current time)
            
    Returns:
        Dict with:
            - timestamp: List[str], hourly timestamps for predictions
            - demand: List[float], predicted demand values
    """
    try:
        # Validate input
        if 'historical_demand' not in input_features:
            raise ValueError("Missing required feature: historical_demand")
            
        historical_demand = np.array(input_features['historical_demand'])
        
        # Load model and scaler (cached)
        model, scaler = load_model()
        
        # Validate historical data length
        if len(historical_demand) != model.input_seq_len:
            raise ValueError(f"Historical demand must contain {model.input_seq_len} hours of data")
        
        # Scale input
        scaled_input = scaler.transform(historical_demand.reshape(-1, 1))
        x = torch.FloatTensor(scaled_input).unsqueeze(0).to(DEVICE)
        
        # Initialize target with zeros
        tgt = torch.zeros((1, model.output_seq_len, 1), device=DEVICE)
        
        # Generate prediction
        with torch.no_grad():
            scaled_pred = model(x, tgt)
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(scaled_pred.cpu().squeeze().numpy().reshape(-1, 1))
        predictions = predictions.flatten()
        
        # Generate timestamps
        start_time = pd.Timestamp(input_features.get('timestamp', pd.Timestamp.now()))
        timestamps = [start_time + timedelta(hours=i) for i in range(len(predictions))]
        
        result = {
            'timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
            'demand': predictions.tolist()
        }
        
        logger.info(f"Successfully generated predictions for {len(predictions)} hours")
        return result
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

# Sample usage
if __name__ == "__main__":
    # Test the prediction function with sample input
    sample_features = {
        'historical_demand': np.random.normal(100, 20, 24).tolist(),  # 24 hours of historical data
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        result = predict_demand(sample_features)
        print("\nPrediction Results:")
        for ts, demand in zip(result['timestamp'][:5], result['demand'][:5]):
            print(f"{ts}: {demand:.2f}")
        print("...")  # Indicate there are more predictions
    except Exception as e:
        print(f"Error: {str(e)}") 