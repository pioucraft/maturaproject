#ifndef POOLING_H 
#define POOLING_H

#include "nn.h"
#include "utils.h"

int create_pooling_layer(Layer* layer, int input_dimensions, int output_dimensions, int pool_dimensions, int channels);

int pooling_forward(Layer layer);

#endif
