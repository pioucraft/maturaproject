#include <stdio.h>

#define DATA_TYPE float

#define POOL_TYPE_MAX 0
#define POOL_TYPE_MEAN 1

typedef struct Convolution_layer {
    int filter_dimensions;
    int filter_stride;
    DATA_TYPE* filter_parameters;
    DATA_TYPE* filter_bias;

    int pool_dimensions;
    int pool_type;
} Convolution_layer;

typedef struct Neuron {
    DATA_TYPE* weights;
    int num_weights;
    DATA_TYPE bias;
} Neuron;

typedef struct MLP_layer {
    Neuron* neurons;
    int num_neurons;
} MLP_layer;

typedef struct CNN {
    Convolution_layer* convolution_layers;
    int num_convolution_layers;

    MLP_layer* mlp_layers;
    int num_mlp_layers;
} CNN;

int main() {
    printf("Hello, CUDA!\n");
    return 0;
}

