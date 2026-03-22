#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <stdio.h>

#include "convolution.h"
#include "nn.h"
#include "utils.h"

int create_convolution_layer(Layer* layer, int input_dimensions, int output_dimensions, int filter_dimensions, int num_filters, int in_channels, int out_channels) {
    DATA_TYPE* filters;
    DATA_TYPE* biases;

    cudaMalloc(&filters, num_filters * filter_dimensions * filter_dimensions * in_channels * sizeof(DATA_TYPE));
    cudaMalloc(&biases, num_filters * sizeof(DATA_TYPE));

    for(int i = 0; i < num_filters * filter_dimensions * filter_dimensions * in_channels; i++) {
        DATA_TYPE filter = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * 0.5 - 0.25);
        cudaMemcpy(filters + i, &filter, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    for(int i = 0; i < num_filters; i++) {
        DATA_TYPE bias = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * 0.5 - 0.25);
        cudaMemcpy(biases + i, &bias, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    DATA_TYPE* filter_grads;
    DATA_TYPE* bias_grads;

    cudaMalloc(&filter_grads, num_filters * filter_dimensions * filter_dimensions * in_channels * sizeof(DATA_TYPE));
    cudaMalloc(&bias_grads, num_filters * sizeof(DATA_TYPE));

    *layer = {
        .layer_type = LAYER_TYPE_CONVOLUTION,
        .num_in_channels = in_channels,
        .num_out_channels = out_channels,
        .input = {
            .d2 = {
                .input_dimensions = input_dimensions 
            }
        },
        .output = {
            .d2 = {
                .output_dimensions = output_dimensions
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
    int filter = blockIdx.x;

    int output_x = threadIdx.x % layer.output.d2.output_dimensions;
    int output_y = threadIdx.x / layer.output.d2.output_dimensions;
    
    int output_channel_offset = output_channel * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions;
    int filter_offset = filter * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions * layer.num_in_channels;

    int output_location = output_channel_offset + output_y * layer.output.d2.output_dimensions + output_x;
    layer.output.d2.output[output_location] = layer.layer.convolution_layer.biases[filter];
    for(int input_channel = 0; input_channel < layer.num_in_channels; input_channel++) {

        int input_channel_offset = input_channel * layer.input.d2.input_dimensions * layer.input.d2.input_dimensions;
        
        for(int x = 0; x < layer.layer.convolution_layer.filter_dimensions; x++) {
            for(int y = 0; y < layer.layer.convolution_layer.filter_dimensions; y++) {
                int input_x = output_x + x;
                int input_y = output_y + y;

                int filter_channel_offset = filter_offset + input_channel * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions;

                DATA_TYPE input_value = layer.input.d2.input[input_channel_offset + input_y * layer.input.d2.input_dimensions + input_x];
                DATA_TYPE filter_value = layer.layer.convolution_layer.filters[filter_channel_offset + y * layer.layer.convolution_layer.filter_dimensions + x];

                layer.output.d2.output[output_location] += input_value * filter_value;
            }
        }
    }

    if(layer.output.d2.output[output_location] < 0) {
        layer.output.d2.output[output_location] = 0;
    }
}


__global__ void zero_grads_convolution_layer(Layer layer) {
    int filter = blockIdx.x;

    if(threadIdx.x == 0) {
        layer.layer.convolution_layer.bias_grads[filter] = (DATA_TYPE)0.0;
    }
    for(int in_channel = 0; in_channel < layer.num_in_channels; in_channel++) {
        int filter_idx = filter * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions * layer.num_in_channels + in_channel * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions + threadIdx.x;
        layer.layer.convolution_layer.filter_grads[filter_idx] = (DATA_TYPE)0.0;
    }
}

__global__ void zero_input_grads_convolution_layer(Layer layer) {
    int input_channel = blockIdx.x;
    int input_idx = input_channel * blockDim.x + threadIdx.x;

    layer.input.d2.grads[input_idx] = (DATA_TYPE)0.0;
}

__global__ void grad_convolution_layer(Layer layer) {
    int output_channel = blockIdx.x;
    int filter = blockIdx.x;

    int output_x = threadIdx.x % layer.output.d2.output_dimensions;
    int output_y = threadIdx.x / layer.output.d2.output_dimensions;
    
    int output_channel_offset = output_channel * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions;
    int filter_offset = filter * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions;

    atomicAdd(&(layer.layer.convolution_layer.bias_grads[filter]), layer.output.d2.grads[output_channel_offset + output_y * layer.output.d2.output_dimensions + output_x]);

    for(int input_channel = 0; input_channel < layer.num_in_channels; input_channel++) {

        int input_channel_offset = input_channel * layer.input.d2.input_dimensions * layer.input.d2.input_dimensions;

        for(int x = 0; x < layer.layer.convolution_layer.filter_dimensions; x++) {
            for(int y = 0; y < layer.layer.convolution_layer.filter_dimensions; y++) {
                int input_x = output_x + x;
                int input_y = output_y + y;

                DATA_TYPE input_value = layer.input.d2.input[input_channel_offset + input_y * layer.input.d2.input_dimensions + input_x];
                DATA_TYPE grad_value = layer.output.d2.grads[output_channel_offset + output_y * layer.output.d2.output_dimensions + output_x];

                int filter_channel_offset = filter_offset + input_channel * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions;

                atomicAdd(&(layer.layer.convolution_layer.filter_grads[filter_channel_offset + y * layer.layer.convolution_layer.filter_dimensions + x]), grad_value * input_value);

                if(layer.input.d2.grads != NULL) {
                    if(input_value > 0) {
                        atomicAdd(&(layer.input.d2.grads[input_channel_offset + input_y * layer.input.d2.input_dimensions + input_x]), grad_value * layer.layer.convolution_layer.filters[filter_channel_offset + y * layer.layer.convolution_layer.filter_dimensions + x]);
                    }
                }

            }
        }
    }


}

__global__ void update_convolution_layer(Layer layer, float learning_rate) {
    int filter = blockIdx.x;

    if(threadIdx.x == 0) {
        layer.layer.convolution_layer.biases[filter] -= learning_rate * layer.layer.convolution_layer.bias_grads[filter];
    }

    for(int in_channel = 0; in_channel < layer.num_in_channels; in_channel++) {
        int filter_idx = filter * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions * layer.num_in_channels + in_channel * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions + threadIdx.x;
        layer.layer.convolution_layer.filters[filter_idx] -= learning_rate * layer.layer.convolution_layer.filter_grads[filter_idx];
    }
}
