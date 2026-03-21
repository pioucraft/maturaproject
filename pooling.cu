#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <stdio.h>

#include "pooling.h"
#include "nn.h"
#include "utils.h"

int create_pooling_layer(Layer* layer, int input_dimensions, int output_dimensions, int pool_dimensions, int channels) {
    *layer = {
        .layer_type = LAYER_TYPE_POOLING,
        .num_in_channels = channels,
        .num_out_channels = channels,
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
            .pooling_layer = {
                .pool_dimensions = pool_dimensions
            }
        }
    };
    return 0;
}

__global__ void pooling_forward(Layer layer) {
    int channel = blockIdx.x;
    int output_x = threadIdx.x % layer.output.d2.output_dimensions;
    int output_y = threadIdx.x / layer.output.d2.output_dimensions;

    int channel_input_offset = channel * layer.input.d2.input_dimensions * layer.input.d2.input_dimensions;
    int channel_output_offset = channel * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions;

    layer.output.d2.output[channel_output_offset + output_y * layer.output.d2.output_dimensions + output_x] = (DATA_TYPE)0.0;
    for(int x = 0; x < layer.layer.pooling_layer.pool_dimensions; x++) {
        for(int y = 0; y < layer.layer.pooling_layer.pool_dimensions; y++) {

            int input_x = output_x * layer.layer.pooling_layer.pool_dimensions + x;
            int input_y = output_y * layer.layer.pooling_layer.pool_dimensions + y;

            DATA_TYPE input_value = layer.input.d2.input[channel_input_offset + input_y * layer.input.d2.input_dimensions + input_x];

            if(input_value > layer.output.d2.output[channel_output_offset + output_y * layer.output.d2.output_dimensions + output_x]) {
                layer.output.d2.output[channel_output_offset + output_y * layer.output.d2.output_dimensions + output_x] = input_value;
            }
        }
    }
}


__global__ void grad_pooling_layer(Layer layer) {
    int channel = blockIdx.x;
    int output_x = threadIdx.x % layer.output.d2.output_dimensions;
    int output_y = threadIdx.x / layer.output.d2.output_dimensions;

    int channel_input_offset = channel * layer.input.d2.input_dimensions * layer.input.d2.input_dimensions;
    int channel_output_offset = channel * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions;


    if(layer.output.d2.grads == NULL) {
        return;
    }
    for(int x = 0; x < layer.layer.pooling_layer.pool_dimensions; x++) {
        for(int y = 0; y < layer.layer.pooling_layer.pool_dimensions; y++) {

            int input_x = output_x * layer.layer.pooling_layer.pool_dimensions + x;
            int input_y = output_y * layer.layer.pooling_layer.pool_dimensions + y;

            DATA_TYPE input_value = layer.input.d2.input[channel_input_offset + input_y * layer.input.d2.input_dimensions + input_x];

            if(input_value == layer.output.d2.output[channel_output_offset + output_y * layer.output.d2.output_dimensions + output_x]) {
                layer.input.d2.grads[channel_input_offset + input_y * layer.input.d2.input_dimensions + input_x] = layer.output.d2.grads[channel_output_offset + output_y * layer.output.d2.output_dimensions + output_x];
            }
        }
    }
}

