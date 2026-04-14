#include "tanh.h"

int create_tanh_layer(Layer* layer, int input_size) {
    *layer = (Layer) {
        .layer_type = LAYER_TYPE_TANH,
        .num_in_channels = 1,
        .num_out_channels = 1,
        .input = {
            .d1 = {
                .input_size = input_size
            }
        },
        .output = {
            .d1 = {
                .output_size = input_size
            }
        },
    };
    return 0;
}
