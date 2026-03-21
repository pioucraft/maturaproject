#ifndef CONVOLUTION_H 
#define CONVOLUTION_H

#include <cuda_runtime.h>

#include "nn.h"
#include "utils.h"


int create_convolution_layer(Layer* layer, int input_dimensions, int output_dimensions, int filter_dimensions, int num_filters, int in_channels, int out_channels);

__global__ void convolution_forward(Layer layer);

__global__ void zero_grads_convolution_layer(Layer layer);

__global__ void grad_convolution_layer(Layer layer);

__global__ void update_convolution_layer(Layer layer, DATA_TYPE learning_rate);

#endif
