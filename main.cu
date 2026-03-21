#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "mlp.h"
#include "mnist.h"
#include "nn.h"
#include "utils.h"

#define BATCH_SIZE 32
#define NUM_CYCLES 100
#define DATASET_SIZE 3000

int main() {
    printf("Hello, CUDA!\n");

    MNIST_Image* dataset;
    load_mnist_dataset("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", &dataset, DATASET_SIZE);

    Layer* layers = (Layer*)malloc(sizeof(*layers) * 3);

    create_mlp_layer(&(layers[0]), 28 * 28, 128);
    create_mlp_layer(&(layers[1]), 128, 128);
    create_mlp_layer(&(layers[2]), 128, 10);

    NN nn = {
        .num_layers = 3,
        .layers = layers
    };

    create_nn(&nn);


    for(int cycle = 0; cycle < NUM_CYCLES; cycle++) {
        printf("Cycle %d\n", cycle);
        for(int i = 0; i < (DATASET_SIZE - BATCH_SIZE); i += BATCH_SIZE) {
            zero_grads_nn(&nn);
            for(int j = 0; j < BATCH_SIZE; j++) {
                call_nn(&nn, dataset[i + j].pixels);
            }
        }
        
        call_nn(&nn, dataset[420].pixels);
        display_nn_output_mnist(&nn, dataset[420].label);

    }


    return 0;
}

