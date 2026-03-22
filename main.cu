#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "convolution.h"
#include "mlp.h"
#include "mnist.h"
#include "nn.h"
#include "pooling.h"
#include "utils.h"

#define BATCH_SIZE 32
#define NUM_CYCLES 100
#define DATASET_SIZE 60000
#define TEST_DATASET_SIZE 10000
#define LEARNING_RATE 1e-4

int main() {
    printf("Hello, CUDA!\n");

    MNIST_Image* dataset;
    load_mnist_dataset("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", &dataset, DATASET_SIZE);

    MNIST_Image* test_dataset;
    load_mnist_dataset("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte", &test_dataset, TEST_DATASET_SIZE);

    Layer* layers = (Layer*)malloc(sizeof(*layers) * 6);

    create_convolution_layer(&(layers[0]), 28, 26, 3, 6, 1, 6);
    create_pooling_layer(&(layers[1]), 26, 13, 2, 6);
    create_convolution_layer(&(layers[2]), 13, 10, 4, 12, 6, 12);
    create_pooling_layer(&(layers[3]), 10, 5, 2, 12);
    create_mlp_layer(&(layers[4]), 5*5*12, 128);
    create_mlp_layer(&(layers[5]), 128, 10);

    NN nn = {
        .num_layers = 6,
        .layers = layers
    };

    create_nn(&nn);


    for(int cycle = 0; cycle < NUM_CYCLES; cycle++) {
        printf("Cycle %d\n", cycle);

        call_nn(&nn, test_dataset[0].pixels);
        display_nn_output_mnist(&nn, test_dataset[0].label);
        int correct_predictions = 0;
        for(int i = 0; i < TEST_DATASET_SIZE; i++) {
            call_nn(&nn, test_dataset[i].pixels);
            DATA_TYPE output[10];
            cudaMemcpy(output, nn.layers[nn.num_layers - 1].output.d1.output, 10 * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

            DATA_TYPE label[10];
            cudaMemcpy(label, test_dataset[i].label, 10 * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

            int predicted_label = 0;
            DATA_TYPE max_output = output[0];
            int correct_label = 0;
            

            for(int j = 0; j < 10; j++) {
                if(output[j] > max_output) {
                    max_output = output[j];
                    predicted_label = j;
                }
                if(label[j] > 0.0f) {
                    correct_label = j;
                }
            }
            if(predicted_label == correct_label) {
                correct_predictions++;
            }
        }
        printf("Test accuracy: %.2f%%\n", (float)correct_predictions / TEST_DATASET_SIZE * 100.0f);

        FILE* accuracy_file = fopen("test_accuracy.data", "a");
        fprintf(accuracy_file, "cycle %d: %.2f%%\n", cycle, (float)correct_predictions / TEST_DATASET_SIZE * 100.0f);
        fclose(accuracy_file);

        for(int i = 0; i < (DATASET_SIZE - BATCH_SIZE); i += BATCH_SIZE) {
            zero_grads_nn(&nn);
            for(int j = 0; j < BATCH_SIZE; j++) {
                if((i + j) % 10000 == 0) {
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

