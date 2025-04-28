#!/bin/bash

# Script to build and run the Transformer model training container
# Mounts data and models directories for training and model persistence

# Build the image
# Uncomment after first run
# docker build -t transformer-model-training -f Dockerfile.transformer .

# Run the container with volume mounts
docker run --rm \
  -v "$(pwd)/data_from_2024:/app/data_from_2024" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/nowcast_train.py:/app/nowcast_train.py" \
  transformer-model-training 