#include "pooling.h"
#include <math.h>

int create_pooling_layer(Layer* layer, int input_dimensions, int output_dimensions, int pool_dimensions, int channels) {
    *layer = (Layer) {
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

int pooling_forward(Layer layer) {
    for(int c = 0; c < layer.num_in_channels; c++) {
        for(int o_x = 0; o_x < layer.output.d2.output_dimensions; o_x++) {
            for(int o_y = 0; o_y < layer.output.d2.output_dimensions; o_y++) {
                DATA_TYPE max_value = -INFINITY;
                for(int p_x = 0; p_x < layer.layer.pooling_layer.pool_dimensions; p_x++) {
                    for(int p_y = 0; p_y < layer.layer.pooling_layer.pool_dimensions; p_y++) {
                        int i_x = o_x * layer.layer.pooling_layer.pool_dimensions + p_x;
                        int i_y = o_y * layer.layer.pooling_layer.pool_dimensions + p_y;
                        DATA_TYPE value = layer.input.d2.input[c * layer.input.d2.input_dimensions * layer.input.d2.input_dimensions + i_y * layer.input.d2.input_dimensions + i_x];
                        if(value > max_value) {
                            max_value = value;
                        }
                    }
                }
                layer.output.d2.output[c * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions + o_y * layer.output.d2.output_dimensions + o_x] = max_value;
            }
        }
    }

    return 0;
}
