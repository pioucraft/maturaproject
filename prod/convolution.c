#include <stdlib.h>

#include "convolution.h"

int create_convolution_layer(Layer* layer, int input_dimensions, int output_dimensions, int filter_dimensions, int num_filters, int in_channels, int out_channels) {
    
    DATA_TYPE* filters;
    DATA_TYPE* biases;

    filters = malloc(num_filters * filter_dimensions * filter_dimensions * in_channels * sizeof(DATA_TYPE));
    biases = malloc(num_filters * sizeof(DATA_TYPE));

    for(int i = 0; i < num_filters * filter_dimensions * filter_dimensions * in_channels; i++) {
        DATA_TYPE filter = (DATA_TYPE)((DATA_TYPE)rand() / (DATA_TYPE)RAND_MAX * 0.5 - 0.25);
        filters[i] = filter;
    }

    for(int i = 0; i < num_filters; i++) {
        DATA_TYPE bias = (DATA_TYPE)((DATA_TYPE)rand() / (DATA_TYPE)RAND_MAX * 0.5 - 0.25);
        biases[i] = bias;
    }

    *layer = (Layer) {
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
            }
        }
    };
    return 0;
}

int load_convolution_layer(Layer* layer, FILE* file) {
    fread(layer->layer.convolution_layer.filters, sizeof(DATA_TYPE), layer->layer.convolution_layer.filters_num * layer->layer.convolution_layer.filter_dimensions * layer->layer.convolution_layer.filter_dimensions * layer->num_in_channels, file);
    fread(layer->layer.convolution_layer.biases, sizeof(DATA_TYPE), layer->layer.convolution_layer.filters_num, file);

    return 0;
}
