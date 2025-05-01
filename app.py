from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timezone
from nowcast_predict import predict_demand_from_features
from forecast_predict import predict_future_demand
import mock_weather_data
import pandas as pd
import logging
import sys
import csv
import os
import time
import requests
import holidays
from dotenv import load_dotenv

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request

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

# Dictionary to store zone to coordinate mapping
data_store = {}
# Add a list to store basic zone info for the dropdown/markers
zone_list_store = []

def load_mapping_data() -> None:
    """
    Load CSV mapping data into a dictionary and list on startup.
    """
    global data_store, zone_list_store
    try:
        with open("zone_lookup_lat_long.csv", mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            temp_data_store = {}
            temp_zone_list = []
            for row in reader:
                try:
                    zone_id = int(row["LocationID"])
                    lat = float(row['Latitude']) # Ensure float
                    lon = float(row['Longitude']) # Ensure float
                    temp_data_store[zone_id] = (lat, lon)
                    temp_zone_list.append({
                        "id": zone_id,
                        "name": row.get("Zone", "Unknown Zone"), # Use Zone name
                        "borough": row.get("Borough", "Unknown Borough"),
                        "lat": lat,
                        "lon": lon
                    })
                except (ValueError, KeyError) as row_err:
                    logger.warning(f"Skipping row due to error: {row_err}. Row: {row}")
            data_store = temp_data_store
            zone_list_store = temp_zone_list
            logger.info(f"Successfully loaded {len(data_store)} zones with coordinates.")
    except FileNotFoundError:
        logger.error("zone_lookup_lat_long.csv not found. Cannot load zone coordinates.")
    except Exception as e:
        logger.exception(f"Error loading zone mapping data: {e}")


app = FastAPI(
    title="NYC Taxi Demand Prediction API",
    description="🚕 Predict demand using LSTM for short-term and XGBoost for long-term forecasting",
    version="1.0.0",
    contact={"name": "MSML650 Computing Systems for Machine Learning Spring 2025"},
    license_info={"name": "MIT"}
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serves the main frontend UI at the root path"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health") 
async def health_check():
    """API health check endpoint."""
    return {
        "status": "okay this works as expected!",
        "message": "NYC Taxi Demand Prediction API is running",
        "models": {
            "lstm": "Short-term predictions (up to 2 weeks)",
            "xgboost": "Long-term predictions (beyond 2 weeks)"
        }
    }

# NEW Endpoint to serve zone data with coordinates
@app.get("/zones-with-coords")
async def get_zones_with_coordinates():
    """Returns a list of zones with their ID, name, borough, lat, and lon."""
    if not zone_list_store:
        # Attempt to load again if empty, or handle error
        load_mapping_data()
        if not zone_list_store:
             raise HTTPException(status_code=500, detail="Zone coordinate data not loaded on server.")
    return zone_list_store


@app.on_event("startup")
async def startup_event():
    """Runs on application startup"""
    load_mapping_data()

class PredictionRequest(BaseModel):
    timestamp: str
    zone_id: int
    historical_features: Optional[Dict[str, List[float]]] = None  # For LSTM model

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
        "is_rush_hour": [],
        # New features for enhanced model
        "is_morning_rush": [],
        "is_evening_rush": [],
        "is_morning": [],
        "is_afternoon": [],
        "is_evening": [],
        "is_night": []
    }
    
    # Day of week one-hot encoding
    for i in range(7):
        features[f"day_{i}"] = []
    
    for weather in weather_data:
        # Extract information
        timestamp_fmt = datetime.fromtimestamp(float(weather["data"][0]["dt"]), tz=timezone.utc)
        hour_of_day = timestamp_fmt.hour
        day_of_week = timestamp_fmt.weekday() # Full name of the day
        is_weekend = day_of_week >= 5  # Saturday(5) or Sunday(6)

        # Check if it's a public holiday (US example)
        us_holidays = holidays.US()
        is_holiday = timestamp_fmt.date() in us_holidays

        # Basic features
        features["temp"].append(float(weather["data"][0]["temp"]))
        features["feels_like"].append(float(weather["data"][0]["feels_like"]))
        features["wind_speed"].append(float(weather["data"][0]["wind_speed"]))
        # 2 - Thunderstorm, 3 - Drizzle, 5 - Rain, 6 - Snow
        features["rain_1h"].append(float(1 if int(str(weather["data"][0]["weather"][0]["id"])[0]) in [2, 3, 5, 6] else 0))

        features["hour_of_day"].append(float(hour_of_day))
        features["day_of_week"].append(float(day_of_week))
        features["is_weekend"].append(float(is_weekend))
        features["is_holiday"].append(float(is_holiday))
        
        # Rush hour features
        # Morning Rush Hour: 7:00 AM – 10:00 AM
        # Evening Rush Hour: 4:00 PM – 7:00 PM
        is_rush_hour = 1 if (hour_of_day >= 7 and hour_of_day <= 10) or (hour_of_day >= 16 and hour_of_day <= 19) else 0
        features["is_rush_hour"].append(float(is_rush_hour))
        
        # New segmented rush hour features
        features["is_morning_rush"].append(float(1 if hour_of_day >= 7 and hour_of_day <= 9 else 0))
        features["is_evening_rush"].append(float(1 if hour_of_day >= 16 and hour_of_day <= 19 else 0))
        
        # Day period features
        features["is_morning"].append(float(1 if hour_of_day >= 6 and hour_of_day <= 11 else 0))
        features["is_afternoon"].append(float(1 if hour_of_day >= 12 and hour_of_day <= 17 else 0))
        features["is_evening"].append(float(1 if hour_of_day >= 18 and hour_of_day <= 23 else 0))
        features["is_night"].append(float(1 if hour_of_day >= 0 and hour_of_day <= 5 else 0))
        
        # One-hot encoding for day of week
        for i in range(7):
            features[f"day_{i}"].append(float(1 if day_of_week == i else 0))

    return features

def should_use_xgboost(timestamp: str) -> bool:
    """
    Determine whether to use XGBoost model based on prediction timeframe.
    Uses XGBoost for predictions more than 2 weeks in the future.
    
    Args:
        timestamp: Prediction timestamp in format "%Y-%m-%d %H:%M:%S"
        
    Returns:
        bool: True if XGBoost should be used, False for LSTM
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
        logger.info(f"Using {'XGBoost' if use_xgboost else 'LSTM'} model based on time difference")
        return use_xgboost
    except Exception as e:
        logger.error(f"Error calculating time difference: {str(e)}")
        return False  # Default to LSTM if there's an error

def validate_historical_features(features: Dict[str, List[float]]) -> None:
    """
    Validate the historical features required by the LSTM model.
    
    Args:
        features: Dictionary of historical features
        
    Raises:
        ValueError: If features are invalid or missing
    """
    required_features = {
        # Basic weather features
        'temp', 'feels_like', 'wind_speed', 'rain_1h',
        
        # Time features
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_holiday', 'is_rush_hour',
        
        # Enhanced time features
        'is_morning_rush', 'is_evening_rush',
        'is_morning', 'is_afternoon', 'is_evening', 'is_night',
        
        # Day of week one-hot encoding
        'day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6',
        
        # Zone interactions
        'zone_hour_interaction', 'zone_day_interaction',
        
        # Weather conditions
        'is_cold', 'is_hot', 'rain_rush_interaction',
        
        # LSTM specific features
        'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',  # Lag features
        'demand_rolling_mean_24h', 'demand_rolling_std_24h',  # Rolling statistics
        'demand_change_1h'  # Change features
    }
    
    # Check if all required features are present
    missing_features = required_features - set(features.keys())
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Check if all features have the same length (24 hours for LSTM)
    lengths = {len(v) for v in features.values()}
    if len(lengths) != 1:
        raise ValueError("All features must have the same length (24 hours)")
    if lengths.pop() < 24:
        raise ValueError("All features must have at least 24 hours of data")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict taxi demand for a specific timestamp and zone.
    Routes to appropriate model based on prediction timeframe:
    - LSTM model for predictions within 2 weeks
    - XGBoost model for predictions beyond 2 weeks

    Args:
        request: JSON with:
            - timestamp: Prediction timestamp (required)
            - zone_id: Zone ID to predict for (required)
            - historical_features: Dictionary of historical features (required for LSTM)

    Returns:
        JSON with:
            - timestamp: Prediction timestamp
            - zone_id: Zone ID
            - demand: Predicted demand value
            - model_used: Name of the model used for prediction
    """
    # Log the raw request body upon entry
    try:
        # Convert request to dictionary early for logging, but use request object for access
        request_data_dict = request.dict()
        logger.info(f"Received prediction request data: {request_data_dict}")
    except Exception as log_err:
        logger.error(f"Error converting request to dict for logging: {log_err}")
        # Continue processing even if logging fails initially

    # Main try block for the entire prediction process
    try:
        # Validate timestamp format
        try:
            prediction_time = datetime.strptime(request.timestamp, "%Y-%m-%d %H:%M:%S")
            logger.info(f"Valid timestamp format: {prediction_time}")
        except ValueError as e:
            error_msg = f"Invalid timestamp format: '{request.timestamp}'. Expected YYYY-MM-DD HH:MM:SS. Error: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Determine which model to use
        use_xgboost = should_use_xgboost(request.timestamp)
        logger.info(f"Model selection: {'XGBoost' if use_xgboost else 'LSTM'}")

        if use_xgboost:
            # --- XGBoost Logic ---
            logger.info(f"Attempting XGBoost prediction for zone {request.zone_id} at {request.timestamp}")
            try:
                # Convert timestamp to pandas Timestamp
                pd_timestamp = pd.Timestamp(request.timestamp)
                logger.info(f"Converted timestamp to pandas Timestamp: {pd_timestamp}")

                # Make prediction
                prediction = predict_future_demand(pd_timestamp, request.zone_id)
                logger.info(f"XGBoost prediction successful: {prediction}")
                model_used = "xgboost"
            except FileNotFoundError as e:
                # Specific error for model file missing
                error_msg = f"XGBoost model file not found: {str(e)}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            except Exception as e:
                # Catch any other error during XGBoost prediction
                error_msg = f"XGBoost prediction error: {str(e)}"
                logger.exception(error_msg) # Log full traceback for prediction errors
                raise HTTPException(status_code=500, detail=f"XGBoost prediction failed: {str(e)}")
            # --- End XGBoost Logic ---

        else:
            # --- LSTM Logic ---
            logger.info("Attempting LSTM prediction")
            if not request.historical_features:
                error_msg = "historical_features is required for LSTM predictions (within 14 days)"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)

            try:
                # Validate historical features
                validate_historical_features(request.historical_features)
                logger.info("Historical features validation successful")

                # Make prediction
                result = predict_demand_from_features(
                    features_dict=request.historical_features,
                    zone_id=request.zone_id
                )

                if not result or "demand" not in result or not result["demand"]:
                    error_msg = "Invalid or empty prediction result from LSTM model"
                    logger.error(f"{error_msg}. Result was: {result}")
                    raise ValueError(error_msg) # Raise ValueError to be caught below

                # Get the first prediction (next hour)
                prediction = result["demand"][0]
                model_used = "lstm"
                logger.info(f"LSTM prediction successful: {prediction}")

            except ValueError as e:
                # Catch validation errors or errors raised during prediction
                error_msg = f"LSTM input validation or prediction error: {str(e)}"
                logger.error(error_msg) # Log as error, traceback might not be needed for validation
                raise HTTPException(status_code=400, detail=error_msg) # Bad request if validation fails
            except Exception as e:
                # Catch any other unexpected error during LSTM prediction
                error_msg = f"LSTM prediction error: {str(e)}"
                logger.exception(error_msg) # Log full traceback
                raise HTTPException(status_code=500, detail=f"LSTM prediction failed: {str(e)}")
            # --- End LSTM Logic ---

        # If prediction succeeded, return the response
        return PredictionResponse(
            timestamp=request.timestamp,
            zone_id=request.zone_id,
            demand=prediction,
            model_used=model_used
        )

    except HTTPException:
        # Re-raise HTTP exceptions directly (like 400 Bad Request)
        raise
    except Exception as e:
        # Catch ANY other unexpected error in the main function structure
        error_msg = f"Unexpected error during /predict endpoint processing: {str(e)}"
        logger.exception(error_msg) # Log the full traceback
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected server error occurred: {str(e)}"
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
    # Log entry
    logger.info(f"Received live features request for zone {zone_id}, dt={dt}, mock={mock}")
    
    # Force mock=True in development or when API key isn't set
    if not API_KEY or os.getenv("ENVIRONMENT") != "production":
        if not mock:
            logger.warning("Forcing mock=True because API_KEY is not set or not in production environment")
            mock = True

    # Main try block for the entire feature fetching process
    try:
        if zone_id not in data_store:
            error_msg = f"Invalid zone ID: {zone_id}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        if dt is None:
            dt = int(time.time())
            logger.info(f"Parameter 'dt' not provided, using current time: {dt}")
        else:
            # Validate dt format/range if provided
            try:
                timestamp = float(dt)
                if timestamp < 0 or timestamp > time.time() + 10**9: # Basic range check
                    raise ValueError("Timestamp out of reasonable range")
                datetime.fromtimestamp(timestamp, tz=timezone.utc) # Check conversion
            except (ValueError, TypeError) as e:
                error_msg = f"Invalid timestamp parameter 'dt': {dt}. Error: {str(e)}"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)

        lat, long = data_store[zone_id]
        # Changed from 12 to 24 hours for LSTM model
        time_window = [int(dt - (i * 3600)) for i in range(23, -1, -1)] # Ensure int for API call

        weather_data = []
        if mock:
            logger.info("Using mock weather data.")
            # Use first 24 samples from mock data instead of just 12
            weather_data = mock_weather_data.MOCK_WEATHER_DATA[:24]
        else:
            # --- Live Weather API Call Logic ---
            logger.info("Fetching live weather data from OpenWeather API.")
            if not API_KEY:
                error_msg = "OpenWeather API key not configured on server."
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)

            try:
                # Only fetch a single current weather data point first as a test
                test_url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={long}&dt={time_window[0]}&units=metric&appid={API_KEY}"
                test_response = requests.request("GET", test_url, headers={}, data={}, timeout=5)
                test_response.raise_for_status()
                logger.info("API test successful, proceeding with full data fetch")
            except Exception as e:
                logger.error(f"API test failed: {str(e)}")
                logger.info("Falling back to mock data due to API test failure")
                weather_data = mock_weather_data.MOCK_WEATHER_DATA[:24]
                mock = True  # Switch to mock mode
            
            if not mock:  # Only execute if we're still in live data mode
                for temp_dt in time_window:
                    try:
                        url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={long}&dt={temp_dt}&units=metric&appid={API_KEY}"
                        logger.debug(f"Requesting weather for dt={temp_dt} from URL: {url}")

                        response = requests.request("GET", url, headers={}, data={}, timeout=10) # Add timeout
                        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                        weather_data.append(response.json())
                        logger.debug(f"Successfully fetched weather for dt={temp_dt}")
                        time.sleep(0.1) # Small delay to avoid hitting rate limits if any

                    except requests.exceptions.RequestException as req_err:
                        error_msg = f"Error fetching weather data for timestamp {temp_dt}: {str(req_err)}"
                        logger.exception(error_msg) # Log traceback for request errors
                        # Fall back to mock data instead of failing
                        logger.info("Falling back to mock data due to API error")
                        weather_data = mock_weather_data.MOCK_WEATHER_DATA[:24]
                        break
                    except Exception as e:
                        # Catch other potential errors during the loop
                        error_msg = f"Unexpected error during weather fetch loop for dt={temp_dt}: {str(e)}"
                        logger.exception(error_msg)
                        # Fall back to mock data instead of failing
                        logger.info("Falling back to mock data due to unexpected error")
                        weather_data = mock_weather_data.MOCK_WEATHER_DATA[:24]
                        break
            # --- End Live Weather API Call Logic ---

        # Generate additional features required by LSTM but not by transformer
        base_features = extract_weather_data(weather_data)
        
        # Add lag features - these would typically be calculated from historical data
        # For simplicity, we'll use mock values or derive from the base features
        base_features["demand_lag_1h"] = [1.0] * len(base_features["temp"])  # Example value
        base_features["demand_lag_24h"] = [1.0] * len(base_features["temp"])  # Example value
        base_features["demand_lag_168h"] = [1.0] * len(base_features["temp"])  # Example value
        
        # Rolling statistics - mock values for demonstration
        base_features["demand_rolling_mean_24h"] = [1.0] * len(base_features["temp"])
        base_features["demand_rolling_std_24h"] = [0.2] * len(base_features["temp"])
        
        # Change features
        base_features["demand_change_1h"] = [0.0] * len(base_features["temp"])

        # Ensure all features have exactly 24 hours of data
        for feature in base_features:
            if len(base_features[feature]) < 24:
                base_features[feature].extend([base_features[feature][-1]] * (24 - len(base_features[feature])))
            elif len(base_features[feature]) > 24:
                base_features[feature] = base_features[feature][:24]
        
        # Add zone interaction features necessary for the improved model
        base_features["zone_hour_interaction"] = [zone_id * h for h in base_features["hour_of_day"]]
        base_features["zone_day_interaction"] = [zone_id * d for d in base_features["day_of_week"]]
        
        # Add weather condition features
        base_features["is_cold"] = [float(1 if t < 40 else 0) for t in base_features["temp"]]
        base_features["is_hot"] = [float(1 if t > 80 else 0) for t in base_features["temp"]]
        
        # Add weather-rush hour interaction
        base_features["rain_rush_interaction"] = [r * rh for r, rh in zip(base_features["rain_1h"], base_features["is_rush_hour"])]

        return LiveFeaturesResponse(
            lat=str(lat),
            long=str(long),
            zone_id=zone_id,
            timestamp=datetime.fromtimestamp(float(dt), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            features=base_features
        )

    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    except Exception as e:
        # Catch ANY other unexpected error in the main function structure
        error_msg = f"Unexpected error during /live-features endpoint processing: {str(e)}"
        logger.exception(error_msg) # Log the full traceback
        # Instead of failing with 500, return mock data
        logger.info("Falling back to mock data due to unexpected error")
        lat, long = data_store.get(zone_id, (40.7128, -74.006)) # Default to NYC coordinates
        
        # Use mock data
        weather_data = mock_weather_data.MOCK_WEATHER_DATA[:24]
        base_features = extract_weather_data(weather_data)
        
        # Add required features
        base_features["demand_lag_1h"] = [1.0] * len(base_features["temp"])
        base_features["demand_lag_24h"] = [1.0] * len(base_features["temp"]) 
        base_features["demand_lag_168h"] = [1.0] * len(base_features["temp"])
        base_features["demand_rolling_mean_24h"] = [1.0] * len(base_features["temp"])
        base_features["demand_rolling_std_24h"] = [0.2] * len(base_features["temp"])
        base_features["demand_change_1h"] = [0.0] * len(base_features["temp"])
        
        # Ensure all features have exactly 24 hours of data
        for feature in base_features:
            if len(base_features[feature]) < 24:
                base_features[feature].extend([base_features[feature][-1]] * (24 - len(base_features[feature])))
            elif len(base_features[feature]) > 24:
                base_features[feature] = base_features[feature][:24]
        
        # Add zone interaction features necessary for the improved model
        base_features["zone_hour_interaction"] = [zone_id * h for h in base_features["hour_of_day"]]
        base_features["zone_day_interaction"] = [zone_id * d for d in base_features["day_of_week"]]
        
        # Add weather condition features
        base_features["is_cold"] = [float(1 if t < 40 else 0) for t in base_features["temp"]]
        base_features["is_hot"] = [float(1 if t > 80 else 0) for t in base_features["temp"]]
        
        # Add weather-rush hour interaction
        base_features["rain_rush_interaction"] = [r * rh for r, rh in zip(base_features["rain_1h"], base_features["is_rush_hour"])]

        return LiveFeaturesResponse(
            lat=str(lat),
            long=str(long),
            zone_id=zone_id,
            timestamp=datetime.fromtimestamp(float(dt or time.time()), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            features=base_features
        )