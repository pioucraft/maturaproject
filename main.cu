#include <stdio.h>
#include <cuda_runtime.h>

#define DATA_TYPE float

#define POOL_TYPE_MAX 0
#define POOL_TYPE_MEAN 1

#define LAYER_TYPE_CONVOLUTION 0
#define LAYER_TYPE_POOLING 1
#define LAYER_TYPE_MLP 2

typedef struct Convolution_layer {
    int output_dimensions;
    int filter_dimensions;
    DATA_TYPE* filter_parameters;
    DATA_TYPE filter_bias;
} Convolution_layer;

typedef struct Pooling_layer {
    int output_dimensions;
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
    };
} Layer;

typedef struct CNN {
    int num_layers;
    Layer** layers;
} CNN;

void checkCudaError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

int create_cnn(CNN* cnn, int input_dimensions, int num_layers, Layer layers[]) {
    cnn->num_layers = num_layers;
    cnn->layers = (Layer**)malloc(num_layers * sizeof(Layer*));
    for (int i = 0; i < num_layers; i++) {
        Layer layer = layers[i];

        if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            cudaMalloc(&(layer.convolution_layer.filter_parameters), layer.convolution_layer.filter_dimensions * layer.convolution_layer.filter_dimensions * sizeof(DATA_TYPE));
        } else if(layer.layer_type == LAYER_TYPE_POOLING) {
            // Nothing to allocate for pooling layer
        } else if(layer.layer_type == LAYER_TYPE_MLP) {
            cudaMalloc(&(layer.mlp_layer.neurons), layer.mlp_layer.num_neurons * sizeof(Neuron));
            cudaDeviceSynchronize();

            int num_input = 0;
            if(i == 0) {
                num_input = input_dimensions * input_dimensions; // Assuming input is square
            } else {
                Layer prev_layer = layers[i - 1];
                if(prev_layer.layer_type == LAYER_TYPE_CONVOLUTION) {
                    num_input = prev_layer.convolution_layer.output_dimensions * prev_layer.convolution_layer.output_dimensions;
                } else if(prev_layer.layer_type == LAYER_TYPE_POOLING) {
                    num_input = prev_layer.pooling_layer.output_dimensions * prev_layer.pooling_layer.output_dimensions;
                } else if(prev_layer.layer_type == LAYER_TYPE_MLP) {
                    num_input = prev_layer.mlp_layer.num_neurons;
                }
            }

            for(int j = 0; j < layer.mlp_layer.num_neurons; j++) {
                Neuron neuron;
                cudaMalloc(&(neuron.weights), num_input * sizeof(DATA_TYPE));
                cudaDeviceSynchronize();
                neuron.num_weights = num_input;
                cudaMemcpy(&(layer.mlp_layer.neurons[j]), &neuron, sizeof(Neuron), cudaMemcpyHostToDevice);
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
            .convolution_layer = {
                .output_dimensions = 26,
                .filter_dimensions = 3 // 28x28 -> 26x26
            }
        },
        {
            .layer_type = LAYER_TYPE_CONVOLUTION,
            .convolution_layer = {
                .output_dimensions = 24,
                .filter_dimensions = 3, // 26x26 -> 24x24
            }
        },
        {
            .layer_type = LAYER_TYPE_POOLING,
            .pooling_layer = {
                .output_dimensions = 12,
                .pool_dimensions = 2, // 24x24 -> 12x12
                .pool_type = POOL_TYPE_MAX
            }
        },
        {
            .layer_type = LAYER_TYPE_POOLING,
            .pooling_layer = {
                .output_dimensions = 6,
                .pool_dimensions = 2, // 12x12 -> 6x6
                .pool_type = POOL_TYPE_MAX
            }
        },
        {
            .layer_type = LAYER_TYPE_MLP,
            .mlp_layer = {
                .num_neurons = 30
            }
        },
        {
            .layer_type = LAYER_TYPE_MLP,
            .mlp_layer = {
                .num_neurons = 30
            }
        },
        {
            .layer_type = LAYER_TYPE_MLP,
            .mlp_layer = {
                .num_neurons = 10
            }
        }
    };

    create_cnn(&cnn, 28, 7, layers);
    checkCudaError();

    return 0;
}

