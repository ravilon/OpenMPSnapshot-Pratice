#pragma once

#include "activation_func.h"

#include <vector>
#include <memory>

class Layer {
public:
    Layer(int input_size, int output_size, const std::string& activation_name);
    Layer(std::ifstream& file);
    ~Layer() = default;

    Layer(Layer&& other) noexcept;
    Layer& operator=(Layer&& other) noexcept;

    Layer(const Layer& other);
    Layer& operator=(const Layer& other);

    size_t get_weights_size() const;
    float* get_weights();

    std::vector<float> forward(std::vector<float>& input);

    void save(std::ofstream& file);
    void print_layer();

private:
    int input_size;
    int output_size;
    Activation activation;
    float(*activation_func)(float);
    std::unique_ptr<float[]> weights; // (input_size + 1) * output_size, bias in the last column
};

class MLP {
public:
    friend class Individual;

    MLP();

    // initialize with predefined layers
    MLP(const std::vector<Layer>& layers);

    // initialize with random weights
    MLP(const int* layer_sizes, int num_layers, const std::string& layer_activation, const std::string& output_activation);

    // initialize with weights from file
    MLP(const std::string& filename);

    std::vector<float> forward(std::vector<float>& input);
    void save(const std::string& filename);
    void print_layers();

private:
    std::vector<Layer> layers;
};