#ifndef NN_H
#define NN_H

#define LAYER_TYPE_MLP 1

#include "utils.h"

int create_nn(NN* nn);

int call_nn(NN* nn, DATA_TYPE* input);

int zero_grads_nn(NN* nn);

int grad_nn(NN* nn, DATA_TYPE* expected_output);

int update_nn(NN* nn, DATA_TYPE learning_rate);

#endif
