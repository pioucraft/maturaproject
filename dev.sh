#!/bin/sh
echo "Building the project..."
nvcc -o main main.cu utils.cu mnist.cu nn.cu mlp.cu pooling.cu convolution.cu relu.cu tanh.cu
echo "Running the project..."
> test_accuracy.data
time ./main
