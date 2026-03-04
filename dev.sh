#!/bin/sh
echo "Building the project..."
nvcc -o main main.cu
echo "Running the project..."
./main
