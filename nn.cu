#include <cuda_runtime.h>

#include "mlp.h"
#include "nn.h"
#include "utils.h"

int create_nn(NN* nn) {
    DATA_TYPE* current_input = NULL;

    for(int i = 0; i < nn->num_layers; i++) {
        Layer* layer = &(nn->layers[i]);
        if(layer->layer_type == LAYER_TYPE_MLP) { // 1d input and 1d output
            layer->input.d1.input = current_input;
            cudaMalloc(&(current_input), layer->output.d1.output_size * sizeof(DATA_TYPE));
            layer->output.d1.output = current_input;
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
