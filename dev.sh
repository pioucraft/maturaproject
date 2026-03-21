#!/bin/sh
echo "Building the project..."
nvcc -o main main.cu utils.cu mnist.cu nn.cu mlp.cu pooling.cu convolution.cu
echo "Running the project..."
./main
