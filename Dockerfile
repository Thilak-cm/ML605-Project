FROM python:3.11-slim

WORKDIR /app

# Copy only the training script and requirements
COPY forecast_train.py .
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Create models directory with correct permissions
RUN mkdir -p /app/models && chmod 777 /app/models

# Command to run the training script
CMD ["python", "forecast_train.py"]