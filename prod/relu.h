#ifndef RELU_H 
#define RELU_H 

#include "nn.h"
#include "utils.h"

int create_relu_layer(Layer* layer, int input_size);

int relu_forward(Layer layer);

#endif
