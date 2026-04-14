#include "pooling.h"

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
