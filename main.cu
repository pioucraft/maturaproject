#include <stdio.h>
#include <cuda_runtime.h>

#define DATA_TYPE float

#define POOL_TYPE_MAX 0
#define POOL_TYPE_MEAN 1

#define LAYER_TYPE_CONVOLUTION 0
#define LAYER_TYPE_POOLING 1
#define LAYER_TYPE_MLP 2

typedef struct Convolution_layer {
    int input_dimensions;
    int filter_dimensions;
    int filter_stride;
    DATA_TYPE* filter_parameters;
    DATA_TYPE filter_bias;
} Convolution_layer;

typedef struct Pooling_layer {
    int input_dimensions;
    int pool_dimensions;
    int pool_type; 
} Pooling_layer;

typedef struct Neuron {
    DATA_TYPE* weights;
    int num_weights;
    DATA_TYPE bias;
} Neuron;

typedef struct MLP_layer {
    Neuron* neurons;
    int num_neurons;
} MLP_layer;

typedef struct Layer {
    int layer_type; 
    union {
        Convolution_layer convolution_layer;
        Pooling_layer pooling_layer;
        MLP_layer mlp_layer;
    } layer;
} Layer;

typedef struct CNN {
    int num_layers;
    Layer** layers;
} CNN;

int create_cnn(CNN* cnn, int input_dimensions, int num_layers, Layer layers[]) {
    cnn->num_layers = num_layers;
    cnn->layers = (Layer**)malloc(num_layers * sizeof(Layer*));
    for (int i = 0; i < num_layers; i++) {
        Layer layer = layers[i];

        if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            // TODO: calculate output dimensions
            cudaMalloc(&(layer.layer.convolution_layer.filter_parameters), layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions * sizeof(DATA_TYPE));
        } else if(layer.layer_type == LAYER_TYPE_POOLING) {
            // Nothing to allocate for pooling layer
            // TODO: calculate output dimensions
        } else if(layer.layer_type == LAYER_TYPE_MLP) {
            // TODO: finish this after the two TODOs above are done
            cudaMalloc(&(layer.layer.mlp_layer.neurons), layer.layer.mlp_layer.num_neurons * sizeof(Neuron));
            cudaDeviceSynchronize();
            for(int j = 0; j < layer.layer.mlp_layer.num_neurons; j++) {
                Neuron neuron;
            }

        }

        cudaMalloc(&(cnn->layers[i]), sizeof(Layer));
        cudaDeviceSynchronize();
        cudaMemcpy(cnn->layers[i], &layer, sizeof(Layer), cudaMemcpyHostToDevice);
    }
    return 0;
}

int main() {
    printf("Hello, CUDA!\n");

    CNN cnn;
    Layer layers[] = {
        {
            .layer_type = LAYER_TYPE_CONVOLUTION,
            .layer.convolution_layer = {
                .filter_dimensions = 3, // 28x28 -> 26x26
                .filter_stride = 1,
            }
        },
        {
            .layer_type = LAYER_TYPE_CONVOLUTION,
            .layer.convolution_layer = {
                .filter_dimensions = 3, // 26x26 -> 24x24
                .filter_stride = 1,
            }
        },
        {
            .layer_type = LAYER_TYPE_POOLING,
            .layer.pooling_layer = {
                .pool_dimensions = 2, // 24x24 -> 12x12
                .pool_type = POOL_TYPE_MAX
            }
        },
        {
            .layer_type = LAYER_TYPE_POOLING,
            .layer.pooling_layer = {
                .pool_dimensions = 2, // 12x12 -> 6x6
                .pool_type = POOL_TYPE_MAX
            }
        },
        {
            .layer_type = LAYER_TYPE_MLP,
            .layer.mlp_layer = {
                .num_neurons = 30
            }
        },
        {
            .layer_type = LAYER_TYPE_MLP,
            .layer.mlp_layer = {
                .num_neurons = 30
            }
        },
        {
            .layer_type = LAYER_TYPE_MLP,
            .layer.mlp_layer = {
                .num_neurons = 10
            }
        }
    };

    return 0;
}

