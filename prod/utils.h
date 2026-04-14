#ifndef UTILS_H
#define UTILS_H 

#define DATA_TYPE float

typedef struct Convolution_Layer {
    int filter_dimensions;
    int filters_num;

    DATA_TYPE* filters;
    DATA_TYPE* biases;
} Convolution_Layer;

typedef struct Pooling_Layer {
    int pool_dimensions;
} Pooling_Layer;

typedef struct MLP_Layer {
    DATA_TYPE* weights;
    DATA_TYPE* biases;
} MLP_Layer;

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
        Pooling_Layer pooling_layer;
        Convolution_Layer convolution_layer;
    } layer;
} Layer;

typedef struct NN {
    int num_layers;
    Layer* layers;
} NN;

#endif
