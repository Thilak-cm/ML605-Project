#!/bin/bash

# Script to build and run the XGBoost model training container
# Mounts data and models directories for training and model persistence

# Build the image
# Uncomment after first run
docker build -t xgbr-model-training -f Dockerfile.xgbr .

# Run the container with volume mounts
docker run --rm \
  -v "$(pwd)/data_from_2024:/app/data_from_2024" \
  -v "$(pwd)/models:/app/models" \
  xgbr-model-training