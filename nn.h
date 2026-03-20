#ifndef NN_H
#define NN_H

#define LAYER_TYPE_MLP 1

#include "utils.h"

typedef struct Layer {
    int layer_type;

    int num_in_channels;
    int num_out_channels;

    union {
        struct {
            int input_size;
            DATA_TYPE* input;
        } d1;
        struct {
            int input_dimensions;
            DATA_TYPE* input;
        } d2;
    } input;

    union {
        struct {
            int output_size;
            DATA_TYPE* output;
        } d1;
        struct {
            int output_dimensions;
            DATA_TYPE* output;
        } d2;
    } output;

    union {
        MLP_Layer mlp_layer;
    }
} Layer;

typedef struct NN {
    int num_layers;
    Layer* layers;
} NN;

#endif
