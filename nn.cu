#include <cuda_runtime.h>
#include <stdio.h>

#include "mlp.h"
#include "nn.h"
#include "pooling.h"
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
        } else if(layer->layer_type == LAYER_TYPE_POOLING) { // 2d input and 2d output
            layer->input.d2.input = current_input;
            layer->input.d2.grads = current_input_grads;

            cudaMalloc(&(current_input), layer->num_out_channels * layer->output.d2.output_dimensions * layer->output.d2.output_dimensions * sizeof(DATA_TYPE));
            cudaMalloc(&(current_input_grads), layer->num_out_channels * layer->output.d2.output_dimensions * layer->output.d2.output_dimensions * sizeof(DATA_TYPE));

            layer->output.d2.output = current_input;
            layer->output.d2.grads = current_input_grads;
        }
    }

    checkCudaError();

    return 0;
}

int call_nn(NN* nn, DATA_TYPE* input) {
    if(nn->layers[0].layer_type == LAYER_TYPE_MLP) { // 1d input and 1d output
        nn->layers[0].input.d1.input = input;
    } else if(nn->layers[0].layer_type == LAYER_TYPE_POOLING) { // 2d input and 2d output
        nn->layers[0].input.d2.input = input;
    }

    for(int i = 0; i < nn->num_layers; i++) {
        Layer layer = nn->layers[i];
        if(layer.layer_type == LAYER_TYPE_MLP) {
            int activation_function = (i == nn->num_layers - 1) ? ACTIVATION_FUNCTION_TANH : ACTIVATION_FUNCTION_RELU;
            mlp_forward<<<layer.output.d1.output_size, layer.input.d1.input_size>>>(layer, activation_function);
            cudaDeviceSynchronize();
        } else if(layer.layer_type == LAYER_TYPE_POOLING) {
            pooling_forward<<<layer.num_out_channels, layer.output.d2.output_dimensions * layer.output.d2.output_dimensions>>>(layer);
            cudaDeviceSynchronize();
        }
    }

    checkCudaError();

    return 0;
}

__global__ void zero_grads_layer_1d_output(Layer layer) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    layer.output.d1.grads[output_idx] = (DATA_TYPE)0.0;
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

__global__ void grad_error(Layer output_layer, DATA_TYPE* expected_output) {
    // We assume that the output layer is always an MLP layer with tanh activation function
    int output_idx = threadIdx.x;
    DATA_TYPE error_grad = 2 * (output_layer.output.d1.output[output_idx] - expected_output[output_idx]);
    DATA_TYPE grad = error_grad * (1 - output_layer.output.d1.output[output_idx] * output_layer.output.d1.output[output_idx]);
    output_layer.output.d1.grads[output_idx] = grad;
}

int grad_nn(NN* nn, DATA_TYPE* expected_output) {
    for(int i = nn->num_layers - 1; i >= 0; i--) {
        Layer layer = nn->layers[i];
        if(i == nn->num_layers - 1) {
            grad_error<<<1, layer.output.d1.output_size>>>(layer, expected_output);
        }
        cudaDeviceSynchronize();

        if(layer.layer_type == LAYER_TYPE_MLP) {
            grad_mlp_layer<<<layer.output.d1.output_size, layer.input.d1.input_size>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_POOLING && i != 0) {
            grad_pooling_layer<<<layer.num_out_channels, layer.output.d2.output_dimensions * layer.output.d2.output_dimensions>>>(layer);
        }
        cudaDeviceSynchronize();
    }

    checkCudaError();

    return 0;
}


int update_nn(NN* nn, DATA_TYPE learning_rate) {
    for(int i = 0; i < nn->num_layers; i++) {
        Layer layer = nn->layers[i];
        if(layer.layer_type == LAYER_TYPE_MLP) {
            update_mlp_layer<<<layer.output.d1.output_size, layer.input.d1.input_size>>>(layer, learning_rate);
        }
    }

    cudaDeviceSynchronize();
    checkCudaError();

    return 0;
}
