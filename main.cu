#include <stdio.h>
#include <cuda_runtime.h>

#include "mnist.h"
#include "nn.h"

int main() {
    printf("Hello, CUDA!\n");

    MNIST_Image* dataset;
    load_mnist_dataset("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", &dataset, 60000);

    NN nn;

    return 0;
}

