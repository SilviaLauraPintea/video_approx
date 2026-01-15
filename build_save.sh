#!/bin/bash
# save-container.sh

IMAGE_NAME=my_cuda_image
CONTAINER_NAME=my_cuda_container
OUTPUT_FILE=my_cuda_container.tar

docker build -t $IMAGE_NAME .
docker create --name $CONTAINER_NAME $IMAGE_NAME
docker export $CONTAINER_NAME -o $OUTPUT_FILE
docker rm $CONTAINER_NAME
