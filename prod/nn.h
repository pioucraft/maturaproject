#ifndef NN_H
#define NN_H

#include "utils.h"

#define LAYER_TYPE_MLP 1
#define LAYER_TYPE_POOLING 2
#define LAYER_TYPE_CONVOLUTION 3
#define LAYER_TYPE_RELU 4
#define LAYER_TYPE_TANH 5

int create_nn(NN* nn);

int load_nn(NN* nn, const char* filename);

int call_nn(NN* nn, DATA_TYPE* input);

#endif
