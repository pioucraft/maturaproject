#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <stdio.h>

#include "mlp.h"
#include "nn.h"
#include "utils.h"

int create_mlp_layer(Layer* layer, int input_size, int output_size) {
    DATA_TYPE* weights;
    DATA_TYPE* biases;

    cudaMalloc(&weights, input_size * output_size * sizeof(DATA_TYPE));
    cudaMalloc(&biases, output_size * sizeof(DATA_TYPE));

    for(int i = 0; i < input_size * output_size; i++) {
        DATA_TYPE weight = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * 0.5 - 0.25);
        cudaMemcpy(weights + i, &weight, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < output_size; i++) {
        DATA_TYPE bias = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * 0.5 - 0.25);
        cudaMemcpy(biases + i, &bias, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    DATA_TYPE* weight_grads;
    DATA_TYPE* bias_grads;

    cudaMalloc(&weight_grads, input_size * output_size * sizeof(DATA_TYPE));
    cudaMalloc(&bias_grads, output_size * sizeof(DATA_TYPE));

    *layer = {
        .layer_type = LAYER_TYPE_MLP,
        .num_in_channels = 1,
        .num_out_channels = 1,
        .input = {
            .d1 = {
                .input_size = input_size
            }
        },
        .output = {
            .d1 = {
                .output_size = output_size
            }
        },
        .layer = {
            .mlp_layer = {
                .weights = weights,
                .biases = biases,

                .weight_grads = weight_grads,
                .bias_grads = bias_grads
            }
        }
    };
    return 0;
}

__global__ void mlp_forward(Layer layer, int activation_function) {
    int neuron_idx = blockIdx.x;
    int input_idx = threadIdx.x;
    int weight_idx = neuron_idx * blockDim.x + threadIdx.x;

    if(input_idx == 0) {
        layer.output.d1.output[neuron_idx] = layer.layer.mlp_layer.biases[neuron_idx];
    }
    __syncthreads();
    atomicAdd(&(layer.output.d1.output[neuron_idx]), layer.input.d1.input[input_idx] * layer.layer.mlp_layer.weights[weight_idx]);
    __syncthreads();
    if(input_idx == 0) {
        if(activation_function == ACTIVATION_FUNCTION_RELU && layer.output.d1.output[neuron_idx] < 0) {
            layer.output.d1.output[neuron_idx] = 0;
        } else if(activation_function == ACTIVATION_FUNCTION_TANH) {
            layer.output.d1.output[neuron_idx] = tanh(layer.output.d1.output[neuron_idx]);
        }
    }
}


__global__ void zero_grads_mlp_layer(Layer layer) {
    int neuron_idx = blockIdx.x;
    int weight_idx = neuron_idx * blockDim.x + threadIdx.x;

    if(threadIdx.x == 0) {
        layer.layer.mlp_layer.bias_grads[neuron_idx] = (DATA_TYPE)0.0;
    }
    layer.layer.mlp_layer.weight_grads[weight_idx] = (DATA_TYPE)0.0;
}

__global__ void zero_input_grads_mlp_layer(Layer layer) {
    layer.input.d1.grads[threadIdx.x] = (DATA_TYPE)0.0;
}

__global__ void grad_mlp_layer(Layer layer) {
    // assume that input and hidden layers always use ReLU activation function
    int neuron_idx = blockIdx.x;
    int input_idx = threadIdx.x;
    int weight_idx = neuron_idx * blockDim.x + threadIdx.x;

    if(threadIdx.x == 0) {
        layer.layer.mlp_layer.bias_grads[neuron_idx] += layer.output.d1.grads[neuron_idx];
    }
    layer.layer.mlp_layer.weight_grads[weight_idx] += layer.output.d1.grads[neuron_idx] * layer.input.d1.input[input_idx];

    if(layer.input.d1.grads != NULL) {
        if(layer.input.d1.input[input_idx] > 0) {
            atomicAdd(&(layer.input.d1.grads[input_idx]), layer.output.d1.grads[neuron_idx] * layer.layer.mlp_layer.weights[weight_idx]);
        }
    }
}

__global__ void update_mlp_layer(Layer layer, DATA_TYPE learning_rate) {
    int neuron_idx = blockIdx.x;
    int input_idx = threadIdx.x;
    int weight_idx = neuron_idx * blockDim.x + threadIdx.x;

    if(threadIdx.x == 0) {
        layer.layer.mlp_layer.biases[neuron_idx] -= learning_rate * layer.layer.mlp_layer.bias_grads[neuron_idx];
    }
    layer.layer.mlp_layer.weights[weight_idx] -= learning_rate * layer.layer.mlp_layer.weight_grads[weight_idx];
}
