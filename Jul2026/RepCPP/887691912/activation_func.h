#pragma once

#include <string>

enum class Activation {
    SIGMOID,
    RELU,
    TANH
};

Activation from_string(const std::string& str);
std::string to_string(Activation activation);
float(*get_func(Activation activation))(float);

float sigmoid(float x);
float relu(float x);