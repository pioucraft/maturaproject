#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "mlp.h"
#include "mnist.h"
#include "nn.h"
#include "pooling.h"
#include "utils.h"

#define BATCH_SIZE 32
#define NUM_CYCLES 100
#define DATASET_SIZE 60000
#define LEARNING_RATE 1e-3

int main() {
    printf("Hello, CUDA!\n");

    MNIST_Image* dataset;
    load_mnist_dataset("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", &dataset, DATASET_SIZE);

    MNIST_Image* test_dataset;
    load_mnist_dataset("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte", &test_dataset, 10000);

    Layer* layers = (Layer*)malloc(sizeof(*layers) * 5);

    create_pooling_layer(&(layers[0]), 28, 28, 1, 1);
    create_pooling_layer(&(layers[1]), 28, 14, 2, 1);
    create_mlp_layer(&(layers[2]), 14*14, 128);
    create_mlp_layer(&(layers[3]), 128, 128);
    create_mlp_layer(&(layers[4]), 128, 10);

    NN nn = {
        .num_layers = 5,
        .layers = layers
    };

    create_nn(&nn);


    for(int cycle = 0; cycle < NUM_CYCLES; cycle++) {
        printf("Cycle %d\n", cycle);

        for(int i = 0; i < 10; i++) {
            call_nn(&nn, test_dataset[i].pixels);
            display_nn_output_mnist(&nn, test_dataset[i].label);
        }

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

