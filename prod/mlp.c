#include <stdlib.h>

#include "utils.h"
#include "nn.h"

#include "mlp.h"

int create_mlp_layer(Layer* layer, int input_size, int output_size) {
    DATA_TYPE* weights;
    DATA_TYPE* biases;

    weights = malloc(input_size * output_size * sizeof(DATA_TYPE));
    biases = malloc(output_size * sizeof(DATA_TYPE));

    for(int i = 0; i < input_size * output_size; i++) {
        DATA_TYPE weight = (DATA_TYPE)((DATA_TYPE)rand() / (DATA_TYPE)RAND_MAX * 0.5 - 0.25);
        weights[i] = weight;
    }

    for(int i = 0; i < output_size; i++) {
        DATA_TYPE bias = (DATA_TYPE)((DATA_TYPE)rand() / (DATA_TYPE)RAND_MAX * 0.5 - 0.25);
        biases[i] = bias;
    }

    *layer = (Layer) {
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
            }
        }
    };

    return 0;
}

int load_mlp_layer(Layer* layer, FILE* file) {
    fread(layer->layer.mlp_layer.weights, sizeof(DATA_TYPE), layer->input.d1.input_size * layer->output.d1.output_size, file);
    fread(layer->layer.mlp_layer.biases, sizeof(DATA_TYPE), layer->output.d1.output_size, file);

    return 0;
}

int mlp_forward(Layer layer) {
    for(int o = 0; o < layer.output.d1.output_size; o++) {
        DATA_TYPE sum = layer.layer.mlp_layer.biases[o];
        for(int i = 0; i < layer.input.d1.input_size; i++) {
            sum += layer.input.d1.input[i] * layer.layer.mlp_layer.weights[i * layer.output.d1.output_size + o];
        }
        layer.output.d1.output[o] = sum;
    }

    return 0;
}
