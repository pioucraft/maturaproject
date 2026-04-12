#include <cuda_runtime.h>
#include <stdio.h>

#include "convolution.h"
#include "mlp.h"
#include "nn.h"
#include "pooling.h"
#include "relu.h"
#include "tanh.h"
#include "utils.h"

int create_nn(NN* nn) {
    DATA_TYPE* current_input = NULL;
    DATA_TYPE* current_input_grads = NULL;

    for(int i = 0; i < nn->num_layers; i++) {
        Layer* layer = &(nn->layers[i]);
        if(layer->layer_type == LAYER_TYPE_MLP || layer->layer_type == LAYER_TYPE_RELU || layer->layer_type == LAYER_TYPE_TANH) { // 1d input and 1d output
            layer->input.d1.input = current_input;
            layer->input.d1.grads = current_input_grads;

            cudaMalloc(&(current_input), layer->num_out_channels * layer->output.d1.output_size * sizeof(DATA_TYPE));
            cudaMalloc(&(current_input_grads), layer->num_out_channels * layer->output.d1.output_size * sizeof(DATA_TYPE));

            layer->output.d1.output = current_input;
            layer->output.d1.grads = current_input_grads;
        } else if(layer->layer_type == LAYER_TYPE_POOLING || layer->layer_type == LAYER_TYPE_CONVOLUTION) { // 2d input and 2d output
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
    if(nn->layers[0].layer_type == LAYER_TYPE_MLP || nn->layers[0].layer_type == LAYER_TYPE_RELU || nn->layers[0].layer_type == LAYER_TYPE_TANH) { // 1d input and 1d output
        nn->layers[0].input.d1.input = input;
    } else if(nn->layers[0].layer_type == LAYER_TYPE_POOLING || nn->layers[0].layer_type == LAYER_TYPE_CONVOLUTION) { // 2d input and 2d output
        nn->layers[0].input.d2.input = input;
    }

    for(int i = 0; i < nn->num_layers; i++) {
        Layer layer = nn->layers[i];
        if(layer.layer_type == LAYER_TYPE_MLP) {
            mlp_forward<<<layer.output.d1.output_size, layer.input.d1.input_size>>>(layer);
            cudaDeviceSynchronize();
        } else if(layer.layer_type == LAYER_TYPE_POOLING) {
            pooling_forward<<<layer.num_out_channels, layer.output.d2.output_dimensions * layer.output.d2.output_dimensions>>>(layer);
            cudaDeviceSynchronize();
        } else if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            convolution_forward<<<layer.num_out_channels, layer.output.d2.output_dimensions * layer.output.d2.output_dimensions>>>(layer);
            cudaDeviceSynchronize();
        } else if(layer.layer_type == LAYER_TYPE_RELU) {
            int num_blocks = layer.output.d1.output_size / 128 + 1;
            relu_forward<<<num_blocks, 128>>>(layer);
            cudaDeviceSynchronize();
        } else if(layer.layer_type == LAYER_TYPE_TANH) {
            tanh_forward<<<1, layer.output.d1.output_size>>>(layer);
            cudaDeviceSynchronize();
        }
    }

    checkCudaError();

    return 0;
}

__global__ void zero_grads_layer_1d_output(Layer layer) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(output_idx < layer.output.d1.output_size) {
        layer.output.d1.grads[output_idx] = (DATA_TYPE)0.0;
    }
}

__global__ void zero_grads_layer_2d_output(Layer layer) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;

    layer.output.d2.grads[output_idx] = (DATA_TYPE)0.0;
}

int zero_grads_nn(NN* nn) {
    for(int i = 0; i < nn->num_layers; i++) {
        Layer layer = nn->layers[i];
        if(layer.layer_type == LAYER_TYPE_MLP || layer.layer_type == LAYER_TYPE_RELU || layer.layer_type == LAYER_TYPE_TANH) { // 1d input and 1d output
            int num_blocks = layer.output.d1.output_size * layer.num_out_channels / 128 + 1;
            zero_grads_layer_1d_output<<<num_blocks, 128>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_POOLING || layer.layer_type == LAYER_TYPE_CONVOLUTION) { // 2d input and 2d output
            zero_grads_layer_2d_output<<<layer.num_out_channels, layer.output.d2.output_dimensions * layer.output.d2.output_dimensions>>>(layer);
        }

        if(layer.layer_type == LAYER_TYPE_MLP) {
            zero_grads_mlp_layer<<<layer.output.d1.output_size, layer.input.d1.input_size>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            zero_grads_convolution_layer<<<layer.layer.convolution_layer.filters_num, layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions>>>(layer);
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
    output_layer.output.d1.grads[output_idx] = error_grad;
}

int grad_nn(NN* nn, DATA_TYPE* expected_output) {
    for(int i = nn->num_layers - 1; i >= 0; i--) {
        Layer layer = nn->layers[i];
        if(i == nn->num_layers - 1) {
            grad_error<<<1, layer.output.d1.output_size>>>(layer, expected_output);
        }
        cudaDeviceSynchronize();

        if(layer.layer_type == LAYER_TYPE_MLP) {
            if(layer.input.d1.grads != NULL) {
                zero_input_grads_mlp_layer<<<1, layer.input.d1.input_size>>>(layer);
                cudaDeviceSynchronize();
            }

            grad_mlp_layer<<<layer.output.d1.output_size, layer.input.d1.input_size>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_POOLING) {
            if(layer.input.d2.grads != NULL) {
                zero_input_grads_pooling_layer<<<layer.num_in_channels, layer.input.d2.input_dimensions * layer.input.d2.input_dimensions>>>(layer);
                cudaDeviceSynchronize();
            }
            grad_pooling_layer<<<layer.num_out_channels, layer.output.d2.output_dimensions * layer.output.d2.output_dimensions>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            if(layer.input.d2.grads != NULL) {
                zero_input_grads_convolution_layer<<<layer.num_in_channels, layer.input.d2.input_dimensions * layer.input.d2.input_dimensions>>>(layer);
                cudaDeviceSynchronize();
            }

            grad_convolution_layer<<<layer.num_out_channels, layer.output.d2.output_dimensions * layer.output.d2.output_dimensions>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_RELU) {
            if(layer.input.d1.grads != NULL) {
                int num_blocks = layer.input.d1.input_size / 128 + 1;
                zero_input_grads_relu_layer<<<num_blocks, 128>>>(layer);
                cudaDeviceSynchronize();
            }
            grad_relu_layer<<<layer.output.d1.output_size, 1>>>(layer);
        } else if(layer.layer_type == LAYER_TYPE_TANH) {
            if(layer.input.d1.grads != NULL) {
                zero_input_grads_tanh_layer<<<1, layer.input.d1.input_size>>>(layer);
                cudaDeviceSynchronize();
            }
            grad_tanh_layer<<<1, layer.output.d1.output_size>>>(layer);
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
        } else if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            update_convolution_layer<<<layer.layer.convolution_layer.filters_num, layer.layer.convolution_layer.filter_dimensions * layer.layer.convolution_layer.filter_dimensions>>>(layer, learning_rate);
        }
    }

    cudaDeviceSynchronize();
    checkCudaError();

    return 0;
}

int save_nn(NN* nn, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if(file == NULL) {
        printf("Error opening file for writing: %s\n", filename);
        return -1;
    }

    for(int i = 0; i < nn->num_layers; i++) {
        Layer layer = nn->layers[i];
        if(layer.layer_type == LAYER_TYPE_MLP) {
            save_mlp_layer(layer, file);
        } else if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            save_convolution_layer(layer, file);
        }
    }

    fclose(file);
    return 0;
}

int load_nn(NN* nn, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if(file == NULL) {
        printf("Error opening file for reading: %s\n", filename);
        return -1;
    }

    for(int i = 0; i < nn->num_layers; i++) {
        Layer* layer = &(nn->layers[i]);
        if(layer->layer_type == LAYER_TYPE_MLP) {
            load_mlp_layer(layer, file);
        } else if(layer->layer_type == LAYER_TYPE_CONVOLUTION) {
            load_convolution_layer(layer, file);
        }
    }

    fclose(file);
    return 0;
}
