#!/bin/bash
# build-and-run.sh

IMAGE_NAME=my_cuda_image
CONTAINER_NAME=my_cuda_container


docker build -t $IMAGE_NAME .
docker run --gpus all --runtime nvidia -it --rm \
  --name $CONTAINER_NAME \
  -v "$PWD/ext_files:/app/ext_files" \
  $IMAGE_NAME bash
