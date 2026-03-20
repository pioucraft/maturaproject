#!/bin/sh
echo "Building the project..."
nvcc -o main main.cu utils.cu mnist.cu
echo "Running the project..."
./main
