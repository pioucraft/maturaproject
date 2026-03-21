#ifndef MAIN_H
#define MAIN_H

#define DATA_TYPE float

#define ACTIVATION_FUNCTION_RELU 1
#define ACTIVATION_FUNCTION_TANH 2

void checkCudaError();

typedef struct Pooling_Layer {
    int pool_dimensions;
} Pooling_Layer;

typedef struct MLP_Layer {
    DATA_TYPE* weights;
    DATA_TYPE* biases;

    DATA_TYPE* weight_grads;
    DATA_TYPE* bias_grads;
} MLP_Layer;

typedef struct Layer {
    int layer_type;

    int num_in_channels;
    int num_out_channels;

    union {
        struct {
            int input_size;
            DATA_TYPE* input;
            DATA_TYPE* grads;
        } d1;
        struct {
            int input_dimensions;
            DATA_TYPE* input;
            DATA_TYPE* grads;
        } d2;
    } input;

    union {
        struct {
            int output_size;
            DATA_TYPE* output;
            DATA_TYPE* grads;
        } d1;
        struct {
            int output_dimensions;
            DATA_TYPE* output;
            DATA_TYPE* grads;
        } d2;
    } output;

    union {
        MLP_Layer mlp_layer;
        Pooling_Layer pooling_layer;
    } layer;
} Layer;

typedef struct NN {
    int num_layers;
    Layer* layers;
} NN;

#endif
