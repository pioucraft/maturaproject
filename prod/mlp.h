#ifndef MLP_H
#define MLP_H

#include "utils.h"
#include <stdio.h>

int create_mlp_layer(Layer* layer, int input_size, int output_size);

int load_mlp_layer(Layer* layer, FILE* file);

#endif
