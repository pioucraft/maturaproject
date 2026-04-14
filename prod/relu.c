#include "relu.h"

int create_relu_layer(Layer* layer, int input_size) {
    *layer = (Layer) {
        .layer_type = LAYER_TYPE_RELU,
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

int relu_forward(Layer layer) {
    for(int i = 0; i < layer.input.d1.input_size; i++) {
        DATA_TYPE value = layer.input.d1.input[i];
        if(value > 0) {
            layer.output.d1.output[i] = value;
        } else {
            layer.output.d1.output[i] = 0;
        }
    }
    return 0;
}
