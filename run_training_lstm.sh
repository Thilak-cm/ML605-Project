#!/bin/bash

# Script to build and run the LSTM model training container
# Mounts data and models directories for training and model persistence

# Build the image
# Uncomment after first run
docker build -t lstm-model-training -f Dockerfile.lstm .

# Run the container with volume mounts
docker run --rm \
  -v "$(pwd)/data_from_2024:/app/data_from_2024" \
  -v "$(pwd)/lstm_train.py:/app/lstm_train.py" \
  -v "$(pwd)/models:/app/models" \
  --memory="4g" \
  --cpus=2 \
  lstm-model-training