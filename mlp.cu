#include <cuda_runtime.h>

#include "mlp.h"
#include "nn.h"
#include "utils.h"

int create_mlp_layer(Layer* layer, int input_size, int output_size) {
    DATA_TYPE* weights;
    DATA_TYPE* biases;

    cudaMalloc(&weights, input_size * output_size * sizeof(DATA_TYPE));
    cudaMalloc(&biases, output_size * sizeof(DATA_TYPE));

    *layer = {
        .layer_type = LAYER_TYPE_MLP,
        .num_in_channels = 1,
        .num_out_channels = 1,
        .input = {
            .d1 = {
                .input_size = input_size
            }
        },
        .output = {
            .d1 = {
                .output_size = output_size
            }
        },
        .layer = {
            .mlp_layer = {
                .weights = weights,
                .biases = biases
            }
        }
    };
    return 0;
}
