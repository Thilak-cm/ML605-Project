from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from predict_demand import predict_demand

app = FastAPI(
    title="NYC Taxi Demand Prediction API",
    description="API for predicting taxi demand in NYC using a transformer model",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    historical_demand: List[float]
    timestamp: Optional[str] = None

class PredictionResponse(BaseModel):
    timestamp: List[str]
    demand: List[float]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict taxi demand for the next 24 hours.
    
    Args:
        request: JSON with:
            - historical_demand: List of last 24 hours of demand values
            - timestamp: Optional prediction start time (default: current time)
            
    Returns:
        JSON with:
            - timestamp: List of hourly timestamps for predictions
            - demand: List of predicted demand values
    """
    try:
        # Convert request to dictionary
        input_features = request.dict()
        
        # If timestamp not provided, use current time
        if not input_features["timestamp"]:
            input_features["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        # Get predictions
        result = predict_demand(input_features)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    """API health check endpoint."""
    return {"status": "ok", "message": "NYC Taxi Demand Prediction API is running"} 