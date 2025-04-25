from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timezone
from nowcast_predict import predict_demand
from forecasting_prediction import predict_future_demand
import mock_weather_data
import pandas as pd
import logging
import traceback
import sys
import csv
import os
import time
import requests
import holidays
from dotenv import load_dotenv

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api_errors.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
logger.info("Current working directory: %s", os.getcwd())
logger.info("Checking if .env exists: %s", os.path.exists('.env'))
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        logger.info("Contents of .env: %s", f.read())

load_dotenv()
logger.info("Environment after load_dotenv:")
logger.info("API_KEY from env: %s", os.getenv('api_key'))

# Get API key with error handling
API_KEY = os.getenv('api_key')
if not API_KEY:
    logger.error("OpenWeather API key not found! Please set 'api_key' in .env file")
    # Don't raise error here to allow mock data to work

# Dictionary to store zone to coordinate mapping
data_store = {}

def load_mapping_data() -> None:
    """
    Load CSV mapping data into a dictionary on startup.
    This prevents from reading csv for each request
    """
    global data_store
    with open("zone_lookup_lat_long.csv", mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        data_store = {int(row["LocationID"]): (row['Latitude'], row['Longitude']) for row in reader}

app = FastAPI(
    title="NYC Taxi Demand Prediction API",
    description="🚕 Predict demand using Transformer for short-term and XGBoost for long-term when weather forecasts are either not available or not reliable",
    version="1.0.0",
    contact={"name": "MSML650 Computing Systems for Machine Learning Spring 2025"},
    license_info={"name": "MIT"}
)

@app.on_event("startup")
async def startup_event():
    """Runs on application startup"""
    load_mapping_data()

class PredictionRequest(BaseModel):
    timestamp: str
    zone_id: int
    historical_features: Optional[Dict[str, List[float]]] = None  # For transformer model

class PredictionResponse(BaseModel):
    timestamp: str
    zone_id: int
    demand: float
    model_used: str

class LiveFeaturesResponse(BaseModel):
    lat: str
    long: str
    zone_id: int
    timestamp: str
    features: Optional[Dict[str, List[float]]]

def extract_weather_data(weather_data):
    features = {
        "temp": [],
        "feels_like": [],
        "wind_speed": [],
        "rain_1h": [],
        "hour_of_day": [],
        "day_of_week": [],
        "is_weekend": [],
        "is_holiday": [],
        "is_rush_hour": []
    }
    for weather in weather_data:
        # Extract information
        timestamp_fmt = datetime.fromtimestamp(float(weather["data"][0]["dt"]), tz=timezone.utc)
        hour_of_day = timestamp_fmt.hour
        day_of_week = timestamp_fmt.weekday() # Full name of the day
        is_weekend = day_of_week >= 5  # Saturday(5) or Sunday(6)

        # Check if it's a public holiday (US example)
        us_holidays = holidays.US()
        is_holiday = timestamp_fmt.date() in us_holidays

        features["temp"].append(float(weather["data"][0]["temp"]))
        features["feels_like"].append(float(weather["data"][0]["feels_like"]))
        features["wind_speed"].append(float(weather["data"][0]["wind_speed"]))
        # 2 - Thunderstorm, 3 - Drizzle, 5 - Rain, 6 - Snow
        features["rain_1h"].append(float(1 if int(str(weather["data"][0]["weather"][0]["id"])[0]) in [2, 3, 5, 6] else 0))

        features["hour_of_day"].append(float(hour_of_day))
        features["day_of_week"].append(float(day_of_week))
        features["is_weekend"].append(float(is_weekend))
        features["is_holiday"].append(float(is_holiday))
        # Morning Rush Hour: 7:00 AM – 10:00 AM
        # Evening Rush Hour: 4:00 PM – 7:00 PM
        features["is_rush_hour"].append(1 if (hour_of_day >= 7 and hour_of_day <= 10) or (hour_of_day >= 16 and hour_of_day <= 19) else 0)

    return features

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

@app.get("/live-features/")
async def live_features(zone_id: int, dt: int = None, mock: bool = True):
    """
    This function retrieves live weather features for a specified zone. 
    It performs validation checks on the zone ID and timestamp, fetches 
    historical weather data from OpenWeather API (or mock data), and 
    returns the extracted features.

    Parameters:
    - zone_id (int): Identifier for the geographic zone.
    - dt (int, optional): Unix timestamp for the requested data (default: current time).
    - mock (bool, optional): If True, returns mock weather data instead of making an API call.

    Returns:
    - LiveFeaturesResponse: A structured object containing location details and extracted features.

    Raises:
    - HTTPException (400): If the zone ID is invalid or the timestamp is not within a valid range.

    Process:
    1. Validate `zone_id`: Check if it exists in `data_store`; if not, raise an error.
    2. Validate `dt`:
       - If `dt` is None, set it to the current Unix timestamp.
       - Ensure the timestamp is numeric and within a reasonable range (past 1970, not too far into the future).
       - Convert the timestamp to a datetime object for verification.
    3. Retrieve the latitude and longitude for the given `zone_id`.
    4. Generate a list of timestamps (past 12 hours in hourly intervals).
    5. Fetch weather data:
       - If `mock` is True, use predefined mock data.
       - Otherwise, send API requests to OpenWeather for each timestamp and retrieve JSON responses.
    6. Extract relevant weather features using `extract_weather_data()`.
    7. Return a structured response containing location details, timestamp, and weather features.
    """
    if zone_id not in data_store:
        error_msg = f"Invalid zone: {str(zone_id)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=400,
            detail=error_msg
            )
    if not dt:
        dt = int(time.time())

    try:
        # Ensure it's a number
        timestamp = float(dt)  
        
        # Check if it falls within a reasonable range (e.g., past 1970, not too far into the future)
        if timestamp < 0 or timestamp > time.time() + 10**9:
            error_msg = f"Invalid timestamp: {str(dt)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=400,
                detail=error_msg
                )

        # Convert to datetime to ensure validity
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except ValueError as e:
        error_msg = f"Invalid timestamp: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=400,
            detail=error_msg
            )

    lat, long = data_store[zone_id]
    time_window = []

    for i in range(11, -1, -1):
        time_window.append(dt - (i * 3600))

    # NOTE: There is a limitation with the Weather API, we have only 1000 API calls per day,
    # so if doing development, please make sure to just use the mock data until ready to fully test.
    # Therefore, you can just use the mock data in the `mock_weather_data.py`, please adjust as needed.
    # You can use query param `mock` and set it to `False` to use the Weather API.
    if mock:
        weather_data = mock_weather_data.MOCK_WEATHER_DATA
    else:
        weather_data = []
        for temp_dt in time_window:
            url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={long}&dt={temp_dt}&units=metric&appid={API_KEY}"

            payload={}
            headers = {}

            response = requests.request("GET", url, headers=headers, data=payload)

            if response.status_code != 200:
                error_msg = f"Please verify API Key"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=400,
                    detail=error_msg
                    )


            weather_data.append(response.json())
            

    features = extract_weather_data(weather_data)

    return LiveFeaturesResponse(
        lat=lat,
        long=long,
        zone_id=zone_id,
        timestamp=datetime.fromtimestamp(float(dt), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        features=features
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