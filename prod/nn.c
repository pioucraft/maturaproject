#include <stdio.h>
#include <stdlib.h>

#include "convolution.h"
#include "mlp.h"
#include "pooling.h"
#include "relu.h"
#include "tanh.h"

#include "utils.h"

#include "nn.h"

int create_nn(NN* nn) {
    DATA_TYPE* current_input = NULL;
    DATA_TYPE* current_input_grads = NULL;

    for(int i = 0; i < nn->num_layers; i++) {
        Layer* layer = &(nn->layers[i]);
        if(layer->layer_type == LAYER_TYPE_MLP || layer->layer_type == LAYER_TYPE_RELU || layer->layer_type == LAYER_TYPE_TANH) { // 1d input and 1d output
            layer->input.d1.input = current_input;

            current_input = malloc(layer->num_out_channels * layer->output.d1.output_size * sizeof(DATA_TYPE));

            layer->output.d1.output = current_input;
        } else if(layer->layer_type == LAYER_TYPE_POOLING || layer->layer_type == LAYER_TYPE_CONVOLUTION) { // 2d input and 2d output
            layer->input.d2.input = current_input;

            current_input = malloc(layer->num_out_channels * layer->output.d2.output_dimensions * layer->output.d2.output_dimensions * sizeof(DATA_TYPE));

            layer->output.d2.output = current_input;
        }
    }

    return 0;
}

int load_nn(NN* nn, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if(file == NULL) {
        printf("Error opening file for reading: %s\n", filename);
        return -1;
    }

    for(int i = 0; i < nn->num_layers; i++) {
        Layer* layer = &(nn->layers[i]);
        if(layer->layer_type == LAYER_TYPE_MLP) {
            load_mlp_layer(layer, file);
        } else if(layer->layer_type == LAYER_TYPE_CONVOLUTION) {
            load_convolution_layer(layer, file);
        }
    }

    fclose(file);

    return 0;
}

int call_nn(NN* nn, DATA_TYPE* input) {
    if(nn->layers[0].layer_type == LAYER_TYPE_MLP || nn->layers[0].layer_type == LAYER_TYPE_RELU || nn->layers[0].layer_type == LAYER_TYPE_TANH) { // 1d input and 1d output
        nn->layers[0].input.d1.input = input;
    } else if(nn->layers[0].layer_type == LAYER_TYPE_POOLING || nn->layers[0].layer_type == LAYER_TYPE_CONVOLUTION) { // 2d input and 2d output
        nn->layers[0].input.d2.input = input;
    }

    for(int i = 0; i < nn->num_layers; i++) {
        Layer layer = nn->layers[i];
        if(layer.layer_type == LAYER_TYPE_MLP) {
            mlp_forward(layer);
        } else if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            convolution_forward(layer);
        } else if(layer.layer_type == LAYER_TYPE_POOLING) {
            pooling_forward(layer);
        } else if(layer.layer_type == LAYER_TYPE_RELU) {
            relu_forward(layer);
        } else if(layer.layer_type == LAYER_TYPE_TANH) {
            tanh_forward(layer);
        }
    }

    return 0;
}
