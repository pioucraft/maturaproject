#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define DATA_TYPE float

#define POOL_TYPE_MAX 0
#define POOL_TYPE_MEAN 1

#define LAYER_TYPE_CONVOLUTION 0
#define LAYER_TYPE_POOLING 1
#define LAYER_TYPE_MLP 2

typedef struct Convolution_layer {
    int output_dimensions;
    int filter_dimensions;
    DATA_TYPE* filter_parameters;
    DATA_TYPE* filter_bias;

    DATA_TYPE* output;
} Convolution_layer;

typedef struct Pooling_layer {
    int output_dimensions;
    int pool_dimensions;
    int pool_type; 
} Pooling_layer;

typedef struct Neuron {
    DATA_TYPE* weights;
    int num_weights;
    DATA_TYPE bias;
} Neuron;

typedef struct MLP_layer {
    Neuron* neurons;
    int num_neurons;
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
    MNIST_Image* images = (MNIST_Image*)malloc(num_images * sizeof(MNIST_Image));

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

    for(int i = 0; i < num_images; i++) {
        MNIST_Image c_image;

        DATA_TYPE c_pixels[28 * 28];
        DATA_TYPE c_label[10];

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
        cudaMemcpy(c_image.pixels, &c_pixels, 28 * 28 * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(c_image.label, &c_label, 10 * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    }

    free(images_buffer);
    free(labels_buffer);
    printf("Loaded %d images and labels into GPU memory.\n", num_images);

    return 0;
}

int create_cnn(CNN* cnn, int input_dimensions, int num_layers, Layer layers[]) {
    cnn->num_layers = num_layers;
    cnn->layers = (Layer*)malloc(num_layers * sizeof(Layer));
    for (int i = 0; i < num_layers; i++) {
        Layer layer = layers[i];

        if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {
            cudaMalloc(&(layer.convolution_layer.filter_parameters), layer.convolution_layer.filter_dimensions * layer.convolution_layer.filter_dimensions * sizeof(DATA_TYPE));
            for(int j = 0; j < layer.convolution_layer.filter_dimensions * layer.convolution_layer.filter_dimensions; j++) {
                DATA_TYPE param = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * 0.5 - 0.25);
                cudaMemcpy(layer.convolution_layer.filter_parameters + j, &param, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
            }
            cudaMalloc(&(layer.convolution_layer.filter_bias), sizeof(DATA_TYPE));
            DATA_TYPE bias = (DATA_TYPE)((DATA_TYPE)rand() / RAND_MAX * 0.5 - 0.25);
            cudaMemcpy(layer.convolution_layer.filter_bias, &bias, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
            cudaMalloc(&(layer.convolution_layer.output), layer.convolution_layer.output_dimensions * layer.convolution_layer.output_dimensions * sizeof(DATA_TYPE));

        } else if(layer.layer_type == LAYER_TYPE_POOLING) {
            // Nothing to allocate for pooling layer
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
                cudaMemcpy(&(layer.mlp_layer.neurons[j]), &neuron, sizeof(Neuron), cudaMemcpyHostToDevice);
            }

        }

        cnn->layers[i] = layer;
    }
    cudaDeviceSynchronize();
    return 0;
}

__global__ void convolution_layer(DATA_TYPE* input, DATA_TYPE* filter_parameters, int filter_dimensions, DATA_TYPE* filter_bias, DATA_TYPE* output) {
    
}

int call_cnn(CNN* cnn, DATA_TYPE* input, int num_inputs) {
    DATA_TYPE* current_input = input;
    int current_input_size = num_inputs;

    for(int i = 0; i < cnn->num_layers; i++) {
        Layer layer = cnn->layers[i];

        if(layer.layer_type == LAYER_TYPE_CONVOLUTION) {

        } else if(layer.layer_type == LAYER_TYPE_POOLING) {
        } else if(layer.layer_type == LAYER_TYPE_MLP) {
        }
    }
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
                .filter_dimensions = 3 // 28x28 -> 26x26
            }
        },
        {
            .layer_type = LAYER_TYPE_CONVOLUTION,
            .convolution_layer = {
                .output_dimensions = 24,
                .filter_dimensions = 3, // 26x26 -> 24x24
            }
        },
        {
            .layer_type = LAYER_TYPE_POOLING,
            .pooling_layer = {
                .output_dimensions = 12,
                .pool_dimensions = 2, // 24x24 -> 12x12
                .pool_type = POOL_TYPE_MAX
            }
        },
        {
            .layer_type = LAYER_TYPE_POOLING,
            .pooling_layer = {
                .output_dimensions = 6,
                .pool_dimensions = 2, // 12x12 -> 6x6
                .pool_type = POOL_TYPE_MAX
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

    create_cnn(&cnn, 28, 7, layers);
    checkCudaError();

    MNIST_Image* dataset;
    load_mnist_dataset("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte", &dataset, 60000);
    checkCudaError();

    call_cnn(&cnn, dataset[0].pixels, 28 * 28);
    checkCudaError();

    return 0;
}

