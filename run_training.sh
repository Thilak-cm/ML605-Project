#!/bin/bash

# Build the image
# docker build -t taxi-demand-model-training .

# Run the container with volume mounts
docker run --rm \
  -v "$(pwd)/data_from_2024:/app/data_from_2024" \
  -v "$(pwd)/models:/app/models" \
  taxi-demand-model-training