#!/bin/sh
echo "Building the project..."
nvcc -o main main.cu utils.cu mnist.cu nn.cu
echo "Running the project..."
./main
