#include "tanh.h"
#include <math.h>

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

int tanh_forward(Layer layer) {
    for(int i = 0; i < layer.input.d1.input_size; i++) {
        DATA_TYPE value = layer.input.d1.input[i];
        layer.output.d1.output[i] = tanhf(value);
    }
    return 0;
}
