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

int convolution_forward(Layer layer) {
    for(int f = 0; f < layer.layer.convolution_layer.filters_num; f++) {
        for(int o_x = 0; o_x < layer.output.d2.output_dimensions; o_x++) {
            for(int o_y = 0; o_y < layer.output.d2.output_dimensions; o_y++) {
                DATA_TYPE sum = layer.layer.convolution_layer.biases[f];
                for(int c = 0; c < layer.num_in_channels; c++) {
                    for(int f_x = 0; f_x < layer.layer.convolution_layer.filter_dimensions; f_x++) {
                        for(int f_y = 0; f_y < layer.layer.convolution_layer.filter_dimensions; f_y++) {
                            int i_x = o_x + f_x;
                            int i_y = o_y + f_y;

                            int input_index = c * layer.input.d2.input_dimensions * layer.input.d2.input_dimensions + i_y * layer.input.d2.input_dimensions + i_x;
                            int filter_index = f * layer.num_in_channels * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions + c * layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions + f_y * layer.layer.convolution_layer.filter_dimensions + f_x;

                            sum += layer.input.d2.input[input_index] * layer.layer.convolution_layer.filters[filter_index];
                        }
                    }
                }
                int output_index = f * layer.output.d2.output_dimensions * layer.output.d2.output_dimensions + o_y * layer.output.d2.output_dimensions + o_x;
                layer.output.d2.output[output_index] = sum;
            }
        }
    }

    return 0;
}
