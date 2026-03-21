#include <cuda_runtime.h>

#include "mlp.h"
#include "nn.h"
#include "utils.h"

int create_nn(NN* nn) {
    DATA_TYPE* current_input = NULL;
    DATA_TYPE* current_input_grads = NULL;

    for(int i = 0; i < nn->num_layers; i++) {
        Layer* layer = &(nn->layers[i]);
        if(layer->layer_type == LAYER_TYPE_MLP) { // 1d input and 1d output
            layer->input.d1.input = current_input;
            layer->input.d1.grads = current_input_grads;

            cudaMalloc(&(current_input), layer->num_out_channels * layer->output.d1.output_size * sizeof(DATA_TYPE));
            cudaMalloc(&(current_input_grads), layer->num_out_channels * layer->output.d1.output_size * sizeof(DATA_TYPE));

            layer->output.d1.output = current_input;
            layer->output.d1.grads = current_input_grads;
        }
    }

    checkCudaError();

    return 0;
}

int call_nn(NN* nn, DATA_TYPE* input) {
    if(nn->layers[0].layer_type == LAYER_TYPE_MLP) { // 1d input and 1d output
        nn->layers[0].input.d1.input = input;
    }

    for(int i = 0; i < nn->num_layers; i++) {
        Layer layer = nn->layers[i];
        if(layer.layer_type == LAYER_TYPE_MLP) {
            int activation_function = (i == nn->num_layers - 1) ? ACTIVATION_FUNCTION_TANH : ACTIVATION_FUNCTION_RELU;
            mlp_forward<<<layer.output.d1.output_size, layer.input.d1.input_size>>>(layer, activation_function);
            cudaDeviceSynchronize();
        }
    }

    checkCudaError();

    return 0;
}

__global__ void zero_grads_layer_1d_output(Layer layer) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    layer.output.d1.grads[output_idx] = 0.0f;
}

int zero_grads_nn(NN* nn) {
    for(int i = 0; i < nn->num_layers; i++) {
        Layer layer = nn->layers[i];
        if(layer.layer_type == LAYER_TYPE_MLP) { // 1d input and 1d output
            zero_grads_layer_1d_output<<<layer.num_out_channels, layer.output.d1.output_size>>>(layer);
        }

        if(layer.layer_type == LAYER_TYPE_MLP) {
            zero_grads_mlp_layer<<<layer.output.d1.output_size, layer.input.d1.input_size>>>(layer);
        }
    }

    cudaDeviceSynchronize();
    checkCudaError();

    return 0;
}
