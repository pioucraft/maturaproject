#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "mlp.h"
#include "mnist.h"
#include "nn.h"
#include "utils.h"

int main() {
    printf("Hello, CUDA!\n");

    MNIST_Image* dataset;
    load_mnist_dataset("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", &dataset, 60000);

    Layer* layers = (Layer*)malloc(sizeof(*layers) * 3);

    create_mlp_layer(&(layers[0]), 28 * 28, 128);
    create_mlp_layer(&(layers[1]), 128, 128);
    create_mlp_layer(&(layers[2]), 128, 10);

    NN nn = {
        .num_layers = 3,
        .layers = layers
    };

    create_nn(&nn);
    call_nn(&nn, dataset[0].pixels);
    display_nn_output_mnist(&nn, dataset[0].label);

    return 0;
}

