#ifndef CONVOLUTION_H 
#define CONVOLUTION_H

#include <stdio.h>

#include "nn.h"
#include "utils.h"

int create_convolution_layer(Layer* layer, int input_dimensions, int output_dimensions, int filter_dimensions, int num_filters, int in_channels, int out_channels);

int load_convolution_layer(Layer* layer, FILE* file);

#endif
