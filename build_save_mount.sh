#!/bin/bash
# save-container.sh

IMAGE_NAME=my_cuda_image
CONTAINER_NAME=my_cuda_container
OUTPUT_FILE=my_cuda_container.tar

PROJECT_DIR=$(dirname "$(readlink -f "$0")")

docker build -t $IMAGE_NAME "$PROJECT_DIR"
docker create --name $CONTAINER_NAME \
  -v "$PROJECT_DIR/ext_files:/app/ext_files" \
  $IMAGE_NAME
docker export $CONTAINER_NAME -o "$OUTPUT_FILE"
docker rm $CONTAINER_NAME
