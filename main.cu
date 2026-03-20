#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define CYCLE_COUNT 100
#define BATCH_SIZE 32

#define DATA_TYPE float

#define POOL_TYPE_MAX 0
#define POOL_TYPE_MEAN 1

#define LAYER_TYPE_NONE 0
#define LAYER_TYPE_CONVOLUTION 1
#define LAYER_TYPE_POOLING 2
#define LAYER_TYPE_MLP 3

typedef struct Convolution_layer {
    int output_dimensions;
    int filter_dimensions;
    int filters_number;
    DATA_TYPE* filter_parameters;
    DATA_TYPE* filter_bias;

    DATA_TYPE* output;
    DATA_TYPE* filter_grads;
    DATA_TYPE* bias_grad;

    int in_channels;
    int out_channels;
} Convolution_layer;

typedef struct Pooling_layer {
    int output_dimensions;
    int pool_dimensions;
    int pool_type; 

    DATA_TYPE* output;
    DATA_TYPE* grads;

    int in_channels;
    int out_channels;
} Pooling_layer;

typedef struct Neuron {
    DATA_TYPE* weights;
    int num_weights;
    DATA_TYPE bias;

    DATA_TYPE* weight_grads;
    DATA_TYPE bias_grad;

    DATA_TYPE grad;
} Neuron;

typedef struct MLP_layer {
    Neuron* neurons;
    int num_neurons;

    DATA_TYPE* output;
} MLP_layer;

typedef struct Layer {
    int layer_type; 
    union {
        Convolution_layer convolution_layer;
        Pooling_layer pooling_layer;
        MLP_layer mlp_layer;
    };
} Layer;

typedef struct CNN {
    int num_layers;
    Layer* layers;
} CNN;

void checkCudaError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

typedef struct MNIST_Image {
    DATA_TYPE* pixels; // size 28*28
    DATA_TYPE* label; // size 10
} MNIST_Image;

int load_mnist_dataset(const char* images_path, const char* labels_path, MNIST_Image** dataset, int num_images) {
    *dataset = (MNIST_Image*)malloc(num_images * sizeof(MNIST_Image));

    FILE* images_file = fopen(images_path, "rb");
    FILE* labels_file = fopen(labels_path, "rb");

    unsigned char buffer[4096];
    int read_bytes = 0;

    int total_bytes_images = 0;
    int total_bytes_labels = 0;
    unsigned char* images_buffer = NULL;
    unsigned char* labels_buffer = NULL;

    while((read_bytes = fread(buffer, sizeof(unsigned char), 4096, images_file)) > 0) {
        total_bytes_images += read_bytes;
        images_buffer = (unsigned char*)realloc(images_buffer, total_bytes_images);
        memcpy(images_buffer + total_bytes_images - read_bytes, buffer, read_bytes);
    }

    while((read_bytes = fread(buffer, sizeof(unsigned char), 4096, labels_file)) > 0) {
        total_bytes_labels += read_bytes;
        labels_buffer = (unsigned char*)realloc(labels_buffer, total_bytes_labels);
        memcpy(labels_buffer + total_bytes_labels - read_bytes, buffer, read_bytes);
    }

    DATA_TYPE* c_pixels = (DATA_TYPE*)malloc(28 * 28 * sizeof(DATA_TYPE));
    DATA_TYPE* c_label = (DATA_TYPE*)malloc(10 * sizeof(DATA_TYPE));

    for(int i = 0; i < num_images; i++) {
        MNIST_Image c_image;

        unsigned char* pixels = images_buffer + 16 + i * 28 * 28;
        for(int j = 0; j < 28; j++) {
            for(int k = 0; k < 28; k++) {
                c_pixels[j * 28 + k] = (DATA_TYPE)((DATA_TYPE)pixels[28 * j + k] / 255.0f);
            }
        }

        int label = labels_buffer[8 + i];
        for(int j = 0; j < 10; j++) {
            c_label[j] = (DATA_TYPE)((j == label) ? 1.0f : -1.0f);
        }

        cudaMalloc(&(c_image.pixels), 28 * 28 * sizeof(DATA_TYPE));
        cudaMalloc(&(c_image.label), 10 * sizeof(DATA_TYPE));
        cudaMemcpy(c_image.pixels, c_pixels, 28 * 28 * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(c_image.label, c_label, 10 * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
        (*dataset)[i] = c_image;
    }

    free(images_buffer);
    free(labels_buffer);
    free(c_pixels);
    free(c_label);
    printf("Loaded %d images and labels into GPU memory.\n", num_images);

    return 0;
}

int create_cnn(CNN* cnn, int input_dimensions, int num_layers, Layer layers[]) {
    cnn->num_layers = num_layers;
    cnn->layers = (Layer*)malloc(num_layers * sizeof(Layer));
    for (int i = 0; i < num_layers; i++) {
        Layer layer = layers[i];

        if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            cudaMalloc(&(layer.convolution_layer.filter_parameters), layer.convolution_layer.filters_number * layer.convolution_layer.filter_dimensions * layer.convolution_layer.filter_dimensions * sizeof(DATA_TYPE));
            for(int j = 0; j < layer.convolution_layer.filter_dimensions * layer.convolution_layer.filter_dimensions * layer.convolution_layer.filters_number; j++) {
                DATA_TYPE param = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * 0.5 - 0.25);
                cudaMemcpy(layer.convolution_layer.filter_parameters + j, &param, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
            }
            cudaMalloc(&(layer.convolution_layer.filter_bias), layer.convolution_layer.filters_number * sizeof(DATA_TYPE));
            for(int j = 0; j < layer.convolution_layer.filters_number; j++) {
                DATA_TYPE bias = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * 0.5 - 0.25);
                cudaMemcpy(layer.convolution_layer.filter_bias + j, &bias, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
            }
            cudaMalloc(&(layer.convolution_layer.output), layer.convolution_layer.out_channels * layer.convolution_layer.output_dimensions * layer.convolution_layer.output_dimensions * sizeof(DATA_TYPE));
            cudaMalloc(&(layer.convolution_layer.filter_grads), layer.convolution_layer.filters_number * layer.convolution_layer.filter_dimensions * layer.convolution_layer.filter_dimensions * sizeof(DATA_TYPE));
            cudaMalloc(&(layer.convolution_layer.bias_grad), layer.convolution_layer.filters_number * sizeof(DATA_TYPE));

        } else if(layer.layer_type == LAYER_TYPE_POOLING) {
            cudaMalloc(&(layer.pooling_layer.output), layer.pooling_layer.out_channels * layer.pooling_layer.output_dimensions * layer.pooling_layer.output_dimensions * sizeof(DATA_TYPE));
            cudaMalloc(&(layer.pooling_layer.grads), layer.pooling_layer.in_channels * layer.pooling_layer.output_dimensions * layer.pooling_layer.output_dimensions * layer.pooling_layer.pool_dimensions * layer.pooling_layer.pool_dimensions * sizeof(DATA_TYPE));
        } else if(layer.layer_type == LAYER_TYPE_MLP) {
            cudaMalloc(&(layer.mlp_layer.neurons), layer.mlp_layer.num_neurons * sizeof(Neuron));

            int num_input = 0;
            if(i == 0) {
                num_input = input_dimensions * input_dimensions; // Assuming input is square
            } else {
                Layer prev_layer = layers[i - 1];
                if(prev_layer.layer_type == LAYER_TYPE_CONVOLUTION) {
                    num_input = prev_layer.convolution_layer.output_dimensions * prev_layer.convolution_layer.output_dimensions;
                } else if(prev_layer.layer_type == LAYER_TYPE_POOLING) {
                    num_input = prev_layer.pooling_layer.output_dimensions * prev_layer.pooling_layer.output_dimensions;
                } else if(prev_layer.layer_type == LAYER_TYPE_MLP) {
                    num_input = prev_layer.mlp_layer.num_neurons;
                }
            }

            for(int j = 0; j < layer.mlp_layer.num_neurons; j++) {
                Neuron neuron;
                cudaMalloc(&(neuron.weights), num_input * sizeof(DATA_TYPE));
                neuron.num_weights = num_input;
                for(int k = 0; k < num_input; k++) {
                    DATA_TYPE weight = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * 0.5 - 0.25);
                    cudaMemcpy(neuron.weights + k, &weight, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
                }
                DATA_TYPE bias = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * 0.5 - 0.25);
                neuron.bias = bias;
                cudaMalloc(&(neuron.weight_grads), num_input * sizeof(DATA_TYPE));
                cudaMemcpy(&(layer.mlp_layer.neurons[j]), &neuron, sizeof(Neuron), cudaMemcpyHostToDevice);
            }
            cudaMalloc(&(layer.mlp_layer.output), layer.mlp_layer.num_neurons * sizeof(DATA_TYPE));
        }

        cnn->layers[i] = layer;
    }
    return 0;
}

__global__ void call_convolution_layer(DATA_TYPE* input, int input_dimensions, DATA_TYPE* filter_parameters, int filter_dimensions, DATA_TYPE* filter_bias, DATA_TYPE* output, int output_dimensions, int in_channels) {
    int output_x = threadIdx.x % output_dimensions;
    int output_y = threadIdx.x / output_dimensions;
    int in_channel = blockIdx.x / (blockDim.x / in_channels);
    int channel = blockIdx.x;
    int filter_index = channel % (blockDim.x / in_channels);

    output[channel * output_dimensions * output_dimensions + output_y * output_dimensions + output_x] = 0.0f;
    for(int i = 0; i < filter_dimensions; i++) {
        for(int j = 0; j < filter_dimensions; j++) {
            int input_x = output_x + i;
            int input_y = output_y + j;
            output[channel * output_dimensions * output_dimensions + output_y * output_dimensions + output_x] += input[in_channel * input_dimensions * input_dimensions + input_y * input_dimensions + input_x] * filter_parameters[j * filter_dimensions + i + filter_index * filter_dimensions * filter_dimensions];
        }
    }
    output[channel * output_dimensions * output_dimensions + output_y * output_dimensions + output_x] += filter_bias[filter_index];
    if(output[channel * output_dimensions * output_dimensions + output_y * output_dimensions + output_x] < 0) {
        output[channel * output_dimensions * output_dimensions + output_y * output_dimensions + output_x] = 0;
    }
}

__global__ void call_pooling_layer(DATA_TYPE* input, int input_dimensions, int pool_dimensions, int pool_type, DATA_TYPE* output, int output_dimensions) {
    int output_x = threadIdx.x % output_dimensions;
    int output_y = threadIdx.x / output_dimensions;
    int channel = blockIdx.x;

    // gonna need to implement POOL_MEAN later...
    DATA_TYPE max_value = -INFINITY;
    for(int i = 0; i < pool_dimensions; i++) {
        for(int j = 0; j < pool_dimensions; j++) {
            int input_x = output_x * pool_dimensions + i;
            int input_y = output_y * pool_dimensions + j;
            if(input[input_y * input_dimensions + input_x] > max_value) {
                max_value = input[input_y * input_dimensions + input_x];
            }
        }
    }
    output[output_y * blockDim.x + output_x] = max_value;
    if(output[output_y * blockDim.x + output_x] < 0) {
        output[output_y * blockDim.x + output_x] = 0;
    }
}

__global__ void call_mlp_layer(DATA_TYPE* input, int input_size, Neuron* neurons, int num_neurons, DATA_TYPE* output, int isLastLayer) {
    int neuron_index = blockIdx.x;
    int weight_index = threadIdx.x;
    
    __shared__ DATA_TYPE partials[1024]; // Assuming max input size is 1024
    partials[weight_index] = input[weight_index] * neurons[neuron_index].weights[weight_index];
    __syncthreads();

    if(threadIdx.x == 0) {
        DATA_TYPE sum = neurons[neuron_index].bias;
        for(int i = 0; i < input_size; i++) {
            sum += partials[i];
        }
        output[neuron_index] = sum;
        if(!isLastLayer && output[neuron_index] < 0) {
            output[neuron_index] = 0;
        } else if(isLastLayer) {
            output[neuron_index] = tanh(output[neuron_index]);
        }
    }
}

__global__ void display_cnn(DATA_TYPE* input, Layer output_layer) {
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            if(input[i * 28 + j] > 0.5f) {
                printf("X");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }
    for(int i = 0; i < 10; i++) {
        printf("%d : %.4f\n", i, output_layer.mlp_layer.output[i]);
    }
    printf("\n");
}

int call_cnn(CNN* cnn, DATA_TYPE* input, int input_dimensions, int display_output) {
    DATA_TYPE* current_input = input;
    int current_input_dimensions = input_dimensions;

    int current_input_size = 0;

    for(int i = 0; i < cnn->num_layers; i++) {
        Layer layer = cnn->layers[i];

        if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            int output_dimensions = layer.convolution_layer.output_dimensions;
            call_convolution_layer<<<layer.convolution_layer.out_channels, output_dimensions * output_dimensions>>>(current_input, current_input_dimensions, layer.convolution_layer.filter_parameters, layer.convolution_layer.filter_dimensions, layer.convolution_layer.filter_bias, layer.convolution_layer.output, output_dimensions, layer.convolution_layer.in_channels);
            cudaDeviceSynchronize();
            checkCudaError();
            current_input = layer.convolution_layer.output;
            current_input_dimensions = layer.convolution_layer.output_dimensions;
        } else if(layer.layer_type == LAYER_TYPE_POOLING) {
            int output_dimensions = layer.pooling_layer.output_dimensions;
            call_pooling_layer<<<layer.pooling_layer.in_channels, current_input_dimensions * current_input_dimensions>>>(current_input, current_input_dimensions, layer.pooling_layer.pool_dimensions, layer.pooling_layer.pool_type, layer.pooling_layer.output, output_dimensions);
            cudaDeviceSynchronize();
            checkCudaError();
            current_input_dimensions = layer.pooling_layer.output_dimensions;
        } else if(layer.layer_type == LAYER_TYPE_MLP) {
                current_input_size = current_input_size == 0 ? current_input_dimensions * current_input_dimensions : current_input_size;
                call_mlp_layer<<<layer.mlp_layer.num_neurons, current_input_size>>>(current_input, current_input_size, layer.mlp_layer.neurons, layer.mlp_layer.num_neurons, layer.mlp_layer.output, i == cnn->num_layers - 1);
                cudaDeviceSynchronize();
                checkCudaError();
                current_input = layer.mlp_layer.output;
                current_input_size = layer.mlp_layer.num_neurons;
        }
    }

    if(display_output) {
        display_cnn<<<1, 1>>>(input, cnn->layers[cnn->num_layers - 1]);
        cudaDeviceSynchronize();
        checkCudaError();
    }
    return 0;
}

__global__ void zero_grads_convolution_layer(DATA_TYPE* filter_grads, DATA_TYPE* bias_grad) {
    int index = threadIdx.x;
    filter_grads[index] = 0.0f;
    if(index == 0) {
        *bias_grad = 0.0f;
    }
}


__global__ void zero_grads_mlp_layer(Neuron* neurons) {
    int neuron_index = blockIdx.x;
    int weight_index = threadIdx.x;

    neurons[neuron_index].weight_grads[weight_index] = 0.0f;
    if(threadIdx.x == 0) {
        neurons[neuron_index].bias_grad = 0.0f;
    }
}

__global__ void zero_grads_pooling_layer(DATA_TYPE* grads) {
    int index = threadIdx.x;
    grads[index] = 0.0f;
}

int zero_grads(CNN* cnn, int input_size) {
    int current_input_size = input_size;

    for(int i = 0; i < cnn->num_layers; i++) {
        Layer layer = cnn->layers[i];

        if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {

            int filter_size = layer.convolution_layer.filter_dimensions * layer.convolution_layer.filter_dimensions;
            zero_grads_convolution_layer<<<1, filter_size>>>(layer.convolution_layer.filter_grads, layer.convolution_layer.bias_grad);
            current_input_size = layer.convolution_layer.output_dimensions * layer.convolution_layer.output_dimensions;

        } else if(layer.layer_type == LAYER_TYPE_POOLING) {
            zero_grads_pooling_layer<<<1, layer.pooling_layer.output_dimensions * layer.pooling_layer.output_dimensions>>>(layer.pooling_layer.grads);
            current_input_size = layer.pooling_layer.output_dimensions * layer.pooling_layer.output_dimensions;
        } else if(layer.layer_type == LAYER_TYPE_MLP) {
            zero_grads_mlp_layer<<<layer.mlp_layer.num_neurons, current_input_size>>>(layer.mlp_layer.neurons);
            current_input_size = layer.mlp_layer.num_neurons;
        }
    }
    cudaDeviceSynchronize();
    checkCudaError();
    return 0;
}

__global__ void grad_mlp_layer(Layer layer, Layer previous_layer, Layer next_layer, DATA_TYPE* label, DATA_TYPE* input) {
    int neuron_index = blockIdx.x;
    int weight_index = threadIdx.x;

    Neuron* neuron = &(layer.mlp_layer.neurons[neuron_index]);
    if(threadIdx.x == 0) {
        if(next_layer.layer_type == LAYER_TYPE_NONE) {
            DATA_TYPE error = layer.mlp_layer.output[neuron_index] - label[neuron_index];
            neuron->grad = 2 * error * (1 - layer.mlp_layer.output[neuron_index] * layer.mlp_layer.output[neuron_index]);
        } else {
            DATA_TYPE sum = 0.0f;
            // Next layer must be MLP layer
            for(int i = 0; i < next_layer.mlp_layer.num_neurons; i++) {
                sum += next_layer.mlp_layer.neurons[i].weights[neuron_index] * next_layer.mlp_layer.neurons[i].grad;
            }
            neuron->grad = layer.mlp_layer.output[neuron_index] > 0 ? sum : 0; // ReLU backward
        }
        neuron->bias_grad += neuron->grad;
    }
    __syncthreads();
    
    if(previous_layer.layer_type == LAYER_TYPE_NONE) {
        neuron->weight_grads[weight_index] += neuron->grad * input[weight_index];
    } else {
        if(previous_layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            neuron->weight_grads[weight_index] += neuron->grad * previous_layer.convolution_layer.output[weight_index];
        } else if(previous_layer.layer_type == LAYER_TYPE_POOLING) {
            neuron->weight_grads[weight_index] += neuron->grad * previous_layer.pooling_layer.output[weight_index];
        } else if(previous_layer.layer_type == LAYER_TYPE_MLP) {
            neuron->weight_grads[weight_index] += neuron->grad * previous_layer.mlp_layer.output[weight_index];
        }
    }
}

__global__ void grad_pooling_layer(Layer layer, Layer next_layer, Layer previous_layer) {
    int output_x = threadIdx.x % layer.pooling_layer.output_dimensions;
    int output_y = threadIdx.x / layer.pooling_layer.output_dimensions;

    for(int i = 0; i < layer.pooling_layer.pool_dimensions; i++) {
        for(int j = 0; j < layer.pooling_layer.pool_dimensions; j++) {
            int input_x = output_x * layer.pooling_layer.pool_dimensions + i;
            int input_y = output_y * layer.pooling_layer.pool_dimensions + j;

            DATA_TYPE grad = 0.0f;
            if(previous_layer.layer_type == LAYER_TYPE_CONVOLUTION) {
                grad = (DATA_TYPE)(layer.pooling_layer.output[output_y * layer.pooling_layer.output_dimensions + output_x] == previous_layer.convolution_layer.output[input_y * previous_layer.convolution_layer.output_dimensions + input_x]);
            } else if(previous_layer.layer_type == LAYER_TYPE_POOLING) {
                grad = (DATA_TYPE)(layer.pooling_layer.output[output_y * layer.pooling_layer.output_dimensions + output_x] == previous_layer.pooling_layer.output[input_y * previous_layer.pooling_layer.output_dimensions + input_x]);
            }

            if(grad == 1.0f) {
                if(next_layer.layer_type == LAYER_TYPE_MLP) {
                    grad = 0.0f;
                    for(int k = 0; k < next_layer.mlp_layer.num_neurons; k++) {
                        grad += next_layer.mlp_layer.neurons[k].weights[output_y * layer.pooling_layer.output_dimensions + output_x] * next_layer.mlp_layer.neurons[k].grad;
                    }
                } else if(next_layer.layer_type == LAYER_TYPE_POOLING) {
                    grad = next_layer.pooling_layer.grads[output_y * layer.pooling_layer.output_dimensions + output_x];
                }
            } 
            layer.pooling_layer.grads[input_y * layer.pooling_layer.output_dimensions + input_x] += grad;
        }
    }
}

__global__ void grad_convolution_layer(Layer layer, Layer next_layer, Layer previous_layer, DATA_TYPE* input) {
    int filter_x = threadIdx.x % layer.convolution_layer.filter_dimensions;
    int filter_y = threadIdx.x / layer.convolution_layer.filter_dimensions;

    for(int i = 0; i < layer.convolution_layer.output_dimensions; i++) {
        for(int j = 0; j < layer.convolution_layer.output_dimensions; j++) {
            DATA_TYPE grad = 0.0f;
        }
    }
}

int grad_cnn(CNN cnn, DATA_TYPE* label, DATA_TYPE* input) {
    for(int i = cnn.num_layers - 1; i >= 0; i--) {
        Layer layer = cnn.layers[i];
        Layer previous_layer = i > 0 ? cnn.layers[i - 1] : (Layer){.layer_type = LAYER_TYPE_NONE};
        Layer next_layer = i < cnn.num_layers - 1 ? cnn.layers[i + 1] : (Layer){.layer_type = LAYER_TYPE_NONE};

        if(layer.layer_type == LAYER_TYPE_MLP) {
            int num_weights = 0;
            if(i == 0) 
                num_weights = 28 * 28;
            if(cnn.layers[i - 1].layer_type == LAYER_TYPE_CONVOLUTION)
                num_weights = cnn.layers[i - 1].convolution_layer.output_dimensions * cnn.layers[i - 1].convolution_layer.output_dimensions;
            else if(cnn.layers[i - 1].layer_type == LAYER_TYPE_POOLING)
                num_weights = cnn.layers[i - 1].pooling_layer.output_dimensions * cnn.layers[i - 1].pooling_layer.output_dimensions;
            else if(cnn.layers[i - 1].layer_type == LAYER_TYPE_MLP)
                num_weights = cnn.layers[i - 1].mlp_layer.num_neurons;

            grad_mlp_layer<<<layer.mlp_layer.num_neurons, num_weights>>>(layer, previous_layer, next_layer, label, input);
            cudaDeviceSynchronize();
            checkCudaError();
        } else if(layer.layer_type == LAYER_TYPE_POOLING) {
            grad_pooling_layer<<<1, layer.pooling_layer.output_dimensions * layer.pooling_layer.output_dimensions>>>(layer, next_layer, previous_layer);
        } else if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            grad_convolution_layer<<<1, layer.convolution_layer.filter_dimensions * layer.convolution_layer.filter_dimensions>>>(layer, next_layer, previous_layer, input);
        }
    }
    return 0;
}

__global__ void update_mlp_layer(Neuron* neurons, float learning_rate) {
    int neuron_index = blockIdx.x;
    int weight_index = threadIdx.x;

    Neuron* neuron = &(neurons[neuron_index]);
    neuron->weights[weight_index] -= learning_rate * neuron->weight_grads[weight_index];
    if(threadIdx.x == 0) {
        neuron->bias -= learning_rate * neuron->bias_grad;
    }
}

int update_cnn(CNN* cnn, float learning_rate) {
    for(int i = 0; i < cnn->num_layers; i++) {
        Layer layer = cnn->layers[i];
        if(layer.layer_type == LAYER_TYPE_MLP) {
            int num_weights = 0;
            if(i == 0)                num_weights = 28 * 28;
            if(cnn->layers[i - 1].layer_type == LAYER_TYPE_CONVOLUTION)
                num_weights = cnn->layers[i - 1].convolution_layer.output_dimensions * cnn->layers[i - 1].convolution_layer.output_dimensions;
            else if(cnn->layers[i - 1].layer_type == LAYER_TYPE_POOLING)
                num_weights = cnn->layers[i - 1].pooling_layer.output_dimensions * cnn->layers[i - 1].pooling_layer.output_dimensions;
            else if(cnn->layers[i - 1].layer_type == LAYER_TYPE_MLP)
                num_weights = cnn->layers[i - 1].mlp_layer.num_neurons;
            update_mlp_layer<<<layer.mlp_layer.num_neurons, num_weights>>>(layer.mlp_layer.neurons, learning_rate);
        }
    }
    cudaDeviceSynchronize();
    checkCudaError();
    return 0;
}


int main() {
    printf("Hello, CUDA!\n");

    CNN cnn;
    Layer layers[] = {
        {
            .layer_type = LAYER_TYPE_CONVOLUTION,
            .convolution_layer = {
                .output_dimensions = 26,
                .filter_dimensions = 3, // 28x28 -> 26x26
                .filters_number = 32,
                .in_channels = 1,
                .out_channels = 32
            }
        },
        {
            .layer_type = LAYER_TYPE_POOLING,
            .pooling_layer = {
                .output_dimensions = 13,
                .pool_dimensions = 2, // 26x26 -> 13x13
                .pool_type = POOL_TYPE_MAX,
                .in_channels = 32,
                .out_channels = 32
            }
        },
        {
            .layer_type = LAYER_TYPE_CONVOLUTION,
            .convolution_layer = {
                .output_dimensions = 12,
                .filter_dimensions = 3, // 13x13 -> 12x12
                .filters_number = 2,
                .in_channels = 32,
                .out_channels = 64
            }
        },
        {
            .layer_type = LAYER_TYPE_POOLING,
            .pooling_layer = {
                .output_dimensions = 13,
                .pool_dimensions = 2, // 26x26 -> 13x13
                .pool_type = POOL_TYPE_MAX,
                .in_channels = 64,
                .out_channels = 64
            }
        },
        {
            .layer_type = LAYER_TYPE_MLP,
            .mlp_layer = {
                .num_neurons = 30
            }
        },
        {
            .layer_type = LAYER_TYPE_MLP,
            .mlp_layer = {
                .num_neurons = 30
            }
        },
        {
            .layer_type = LAYER_TYPE_MLP,
            .mlp_layer = {
                .num_neurons = 10
            }
        }
    };

    create_cnn(&cnn, 28, 4, layers);
    checkCudaError();

    MNIST_Image* dataset;
    load_mnist_dataset("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", &dataset, 60000);
    checkCudaError();

    for(int i = 0; i < CYCLE_COUNT; i++) {
        printf("Cycle %d\n", i);
        call_cnn(&cnn, dataset[59999].pixels, 28, 1);
        for(int j = 0; j < (30000 - BATCH_SIZE); j += BATCH_SIZE) {
            zero_grads(&cnn, 28 * 28);
            for(int k = 0; k < BATCH_SIZE; k++) {
                int index = j + k;
                call_cnn(&cnn, dataset[index].pixels, 28, 0);
                grad_cnn(cnn, dataset[index].label, dataset[index].pixels);
                update_cnn(&cnn, 1e-4f);
            }
        }
    }

    return 0;
}

