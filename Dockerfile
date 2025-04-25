FROM python:3.11-slim

WORKDIR /app

# Copy only the training script and requirements
COPY xgbr_forecast_training.py .
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Command to run the training script
CMD ["python", "xgbr_forecast_training.py"]