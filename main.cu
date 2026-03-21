#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "mlp.h"
#include "mnist.h"
#include "nn.h"
#include "utils.h"

#define BATCH_SIZE 32
#define NUM_CYCLES 100
#define DATASET_SIZE 60000
#define LEARNING_RATE 1e-3

int main() {
    printf("Hello, CUDA!\n");

    MNIST_Image* dataset;
    load_mnist_dataset("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", &dataset, DATASET_SIZE);

    Layer* layers = (Layer*)malloc(sizeof(*layers) * 4);

    create_mlp_layer(&(layers[0]), 28 * 28, 128);
    create_mlp_layer(&(layers[1]), 128, 128);
    create_mlp_layer(&(layers[2]), 128, 128);
    create_mlp_layer(&(layers[3]), 128, 10);

    NN nn = {
        .num_layers = 4,
        .layers = layers
    };

    create_nn(&nn);


    for(int cycle = 0; cycle < NUM_CYCLES; cycle++) {
        printf("Cycle %d\n", cycle);

        call_nn(&nn, dataset[0].pixels);
        display_nn_output_mnist(&nn, dataset[0].label);

        for(int i = 0; i < (DATASET_SIZE - BATCH_SIZE); i += BATCH_SIZE) {
            zero_grads_nn(&nn);
            for(int j = 0; j < BATCH_SIZE; j++) {
                if((i + j) % 1000 == 0) {
                    printf("Processing image %d\n", i + j);
                }
                call_nn(&nn, dataset[i + j].pixels);
                grad_nn(&nn, dataset[i + j].label);
            }
            update_nn(&nn, LEARNING_RATE);
        }
        
    }


    return 0;
}

