from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from nowcast_predict import predict_demand
from forecasting_prediction import predict_future_demand
import pandas as pd
import logging
import traceback
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api_errors.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NYC Taxi Demand Prediction API",
    description="🚕 Predict demand using Transformer (short-term) and XGBoost (long-term)",
    version="1.0.0",
    contact={"name": "Thilak Mohan"},
    license_info={"name": "MIT"}
)

class PredictionRequest(BaseModel):
    timestamp: str
    zone_id: int
    historical_features: Optional[Dict[str, List[float]]] = None  # For transformer model

class PredictionResponse(BaseModel):
    timestamp: str
    zone_id: int
    demand: float
    model_used: str

def should_use_xgboost(timestamp: str) -> bool:
    """
    Determine whether to use XGBoost model based on prediction timeframe.
    Uses XGBoost for predictions more than 2 weeks in the future.
    
    Args:
        timestamp: Prediction timestamp in format "%Y-%m-%d %H:%M:%S"
        
    Returns:
        bool: True if XGBoost should be used, False for transformer
    """
    try:
        prediction_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        current_time = datetime.now()
        
        # Calculate time difference in days
        time_difference = (prediction_time - current_time).total_seconds() / (24 * 3600)
        
        # Log the calculation details
        logger.info(f"Current time: {current_time}")
        logger.info(f"Prediction time: {prediction_time}")
        logger.info(f"Time difference: {time_difference} days")
        
        # Use XGBoost if prediction is more than 14 days in the future
        use_xgboost = time_difference > 14
        logger.info(f"Using {'XGBoost' if use_xgboost else 'Transformer'} model based on time difference")
        return use_xgboost
    except Exception as e:
        logger.error(f"Error calculating time difference: {str(e)}")
        return False  # Default to transformer if there's an error

def validate_historical_features(features: Dict[str, List[float]]) -> None:
    """
    Validate the historical features required by the transformer model.
    
    Args:
        features: Dictionary of historical features
        
    Raises:
        ValueError: If features are invalid or missing
    """
    required_features = {
        'temp', 'feels_like', 'wind_speed', 'rain_1h',
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_holiday', 'is_rush_hour'
    }
    
    # Check if all required features are present
    missing_features = required_features - set(features.keys())
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Check if all features have the same length (12 hours)
    lengths = {len(v) for v in features.values()}
    if len(lengths) != 1:
        raise ValueError("All features must have the same length (12 hours)")
    if lengths.pop() != 12:
        raise ValueError("All features must have exactly 12 hours of data")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict taxi demand for a specific timestamp and zone.
    Routes to appropriate model based on prediction timeframe:
    - Transformer model for predictions within 2 weeks
    - XGBoost model for predictions beyond 2 weeks
    
    Args:
        request: JSON with:
            - timestamp: Prediction timestamp (required)
            - zone_id: Zone ID to predict for (required)
            - historical_features: Dictionary of historical features (required for transformer)
            
    Returns:
        JSON with:
            - timestamp: Prediction timestamp
            - zone_id: Zone ID
            - demand: Predicted demand value
            - model_used: Name of the model used for prediction
    """
    try:
        # Convert request to dictionary
        input_features = request.dict()
        logger.info(f"Received prediction request: {input_features}")
        
        # Validate timestamp format
        try:
            prediction_time = datetime.strptime(input_features["timestamp"], "%Y-%m-%d %H:%M:%S")
            logger.info(f"Valid timestamp format: {prediction_time}")
        except ValueError as e:
            error_msg = f"Invalid timestamp format: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
        
        # Determine which model to use
        use_xgboost = should_use_xgboost(input_features["timestamp"])
        logger.info(f"Model selection: {'XGBoost' if use_xgboost else 'Transformer'}")
        
        if use_xgboost:
            # Use XGBoost for long-term predictions
            if not input_features.get("zone_id"):
                error_msg = "zone_id is required for XGBoost predictions"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=400,
                    detail=error_msg
                )
            
            try:
                logger.info(f"Attempting XGBoost prediction for zone {input_features['zone_id']} at {input_features['timestamp']}")
                # Convert timestamp to pandas Timestamp
                pd_timestamp = pd.Timestamp(input_features["timestamp"])
                logger.info(f"Converted timestamp to pandas Timestamp: {pd_timestamp}")
                
                # Make prediction
                prediction = predict_future_demand(
                    pd_timestamp,
                    input_features["zone_id"]
                )
                logger.info(f"XGBoost prediction successful: {prediction}")
                model_used = "xgboost"
            except FileNotFoundError as e:
                error_msg = f"XGBoost model file not found: {str(e)}"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=500,
                    detail=error_msg
                )
            except Exception as e:
                error_msg = f"XGBoost prediction error: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=500,
                    detail=f"XGBoost prediction failed: {str(e)}"
                )
            
        else:
            # Use transformer for short-term predictions
            if not input_features.get("historical_features"):
                error_msg = "historical_features is required for transformer predictions"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=400,
                    detail=error_msg
                )
            
            try:
                # Validate historical features
                validate_historical_features(input_features["historical_features"])
                logger.info("Historical features validation successful")
                
                # Make prediction
                logger.info("Attempting transformer prediction")
                result = predict_demand(
                    zone_id=input_features["zone_id"],
                    historical_features=input_features["historical_features"],
                    timestamp=input_features["timestamp"]
                )
                
                if not result or "demand" not in result:
                    error_msg = "Invalid prediction result from transformer model"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                prediction = result["demand"][0]  # Get first prediction
                model_used = "transformer"
                logger.info(f"Transformer prediction successful: {prediction}")
                
            except ValueError as e:
                error_msg = f"Transformer validation error: {str(e)}"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=400,
                    detail=error_msg
                )
            except Exception as e:
                error_msg = f"Transformer prediction error: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=500,
                    detail=f"Transformer prediction failed: {str(e)}"
                )
        
        return PredictionResponse(
            timestamp=input_features["timestamp"],
            zone_id=input_features["zone_id"],
            demand=prediction,
            model_used=model_used
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.get("/")
async def root():
    """API health check endpoint."""
    return {
        "status": "ok",
        "message": "NYC Taxi Demand Prediction API is running",
        "models": {
            "transformer": "Short-term predictions (up to 2 weeks)",
            "xgboost": "Long-term predictions (beyond 2 weeks)"
        }
    } 