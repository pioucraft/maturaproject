#ifndef MLP_H
#define MLP_H

#include <cuda_runtime.h>

#include "nn.h"
#include "utils.h"

int create_mlp_layer(Layer* layer, int input_size, int output_size);

__global__ void mlp_forward(Layer layer, int activation_function);

#endif
