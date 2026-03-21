#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <stdio.h>

#include "convolution.h"
#include "mlp.h"
#include "nn.h"
#include "utils.h"

int create_convolution_layer(Layer* layer, int input_size, int output_size, int filter_dimensions, int num_filters, int in_channels, int out_channels) {
    DATA_TYPE* filters;
    DATA_TYPE* biases;

    cudaMalloc(&filters, num_filters * filter_dimensions * filter_dimensions * sizeof(DATA_TYPE));
    cudaMalloc(&biases, num_filters * sizeof(DATA_TYPE));

    for(int i = 0; i < num_filters * filter_dimensions * filter_dimensions; i++) {
        DATA_TYPE filter = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * 0.5 - 0.25);
        cudaMemcpy(filters + i, &filter, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < num_filters; i++) {
        DATA_TYPE bias = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * 0.5 - 0.25);
        cudaMemcpy(biases + i, &bias, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    DATA_TYPE* filter_grads;
    DATA_TYPE* bias_grads;

    cudaMalloc(&filter_grads, num_filters * filter_dimensions *filter_dimensions * sizeof(DATA_TYPE));
    cudaMalloc(&bias_grads, num_filters * sizeof(DATA_TYPE));

    *layer = {
        .layer_type = LAYER_TYPE_CONVOLUTION,
        .num_in_channels = in_channels,
        .num_out_channels = out_channels,
        .input = {
            .d2 = {
                .input_dimensions = input_size
            }
        },
        .output = {
            .d2 = {
                .output_dimensions = output_size
            }
        },
        .layer = {
            .convolution_layer = {
                .filter_dimensions = filter_dimensions,
                .filters_num = num_filters,
                .filters = filters,
                .biases = biases,
                .filter_grads = filter_grads,
                .bias_grads = bias_grads
            }
        }
    };
    return 0;
}


__global__ void convolution_forward(Layer layer) {
    int output_channel = blockIdx.x;
    int input_channel = output_channel % layer.num_in_channels;
    int filter = output_channel / layer.num_in_channels;

    int output_x = threadIdx.x % layer.output.d2.output_dimensions;
    int output_y = threadIdx.x / layer.output.d2.output_dimensions;
    
    int output_channel_offset = output_channel * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions;
    int input_channel_offset = input_channel * layer.input.d2.input_dimensions * layer.input.d2.input_dimensions;
    int filter_channel_offset = filter * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions;

    int output_location = output_channel_offset + output_y * layer.layer.convolution_layer.filter_dimensions + output_x;
    layer.output.d2.output[output_location] = layer.layer.convolution_layer.biases[filter];
    for(int x = 0; x < layer.layer.convolution_layer.filter_dimensions; x++) {
        for(int y = 0; y < layer.layer.convolution_layer.filter_dimensions; y++) {
            int input_x = output_x + x;
            int input_y = output_y + y;

            DATA_TYPE input_value = layer.input.d2.input[input_channel_offset + input_y * layer.input.d2.input_dimensions + input_x];
            DATA_TYPE filter_value = layer.layer.convolution_layer.filters[filter_channel_offset + y * layer.layer.convolution_layer.filter_dimensions + x];

            layer.output.d2.output[output_location] += input_value * filter_value;
        }
    }

    __syncthreads();

    if(layer.output.d2.output[output_location] < 0) {
        layer.output.d2.output[output_location] = 0;
    }
}


__global__ void zero_grads_convolution_layer(Layer layer) {
    int filter = blockIdx.x;
    int filter_idx = blockIdx.x * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions + threadIdx.x;

    if(threadIdx.x == 0) {
        layer.layer.convolution_layer.bias_grads[filter] = (DATA_TYPE)0.0;
    }
    layer.layer.convolution_layer.filter_grads[filter_idx] = (DATA_TYPE)0.0;
}

__global__ void grad_convolution_layer(Layer layer) {
    int output_channel = blockIdx.x;
    int input_channel = output_channel % layer.num_in_channels;
    int filter = output_channel / layer.num_in_channels;

    int output_x = threadIdx.x % layer.output.d2.output_dimensions;
    int output_y = threadIdx.x / layer.output.d2.output_dimensions;
    
    int output_channel_offset = output_channel * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions;
    int input_channel_offset = input_channel * layer.input.d2.input_dimensions * layer.input.d2.input_dimensions;
    int filter_channel_offset = filter * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions;

    atomicAdd(&(layer.layer.convolution_layer.bias_grads[filter]), layer.output.d2.grads[output_channel_offset + output_y * layer.output.d2.output_dimensions + output_x]);

    for(int x = 0; x < layer.layer.convolution_layer.filter_dimensions; x++) {
        for(int y = 0; y < layer.layer.convolution_layer.filter_dimensions; y++) {
            int input_x = output_x + x;
            int input_y = output_y + y;

            DATA_TYPE input_value = layer.input.d2.input[input_channel_offset + input_y * layer.input.d2.input_dimensions + input_x];
            DATA_TYPE grad_value = layer.output.d2.grads[output_channel_offset + output_y * layer.output.d2.output_dimensions + output_x];

            atomicAdd(&(layer.layer.convolution_layer.filter_grads[filter_channel_offset + y * layer.layer.convolution_layer.filter_dimensions + x]), grad_value * input_value);

        }
    }


}

__global__ void update_convolution_layer(Layer layer, float learning_rate) {
    int filter = blockIdx.x;
    int filter_idx = blockIdx.x * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions + threadIdx.x;

    if(threadIdx.x == 0) {
        layer.layer.convolution_layer.biases[filter] -= learning_rate * layer.layer.convolution_layer.bias_grads[filter];
    }
    layer.layer.convolution_layer.filters[filter_idx] -= learning_rate * layer.layer.convolution_layer.filter_grads[filter_idx];
}
