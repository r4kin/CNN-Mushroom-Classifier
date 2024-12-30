#include "../../include/cnn/fully_connected.h"
#include <random>

FullyConnected::FullyConnected(int input_size, int output_size) 
    : weights(input_size, output_size), bias(1, output_size) {
    initializeWeights();
}

void FullyConnected::initializeWeights() {
    // Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    double limit = sqrt(6.0 / (weights.get_rows() + weights.get_cols()));
    std::uniform_real_distribution<double> dis(-limit, limit);
    
    for(int i = 0; i < weights.get_rows(); i++) {
        for(int j = 0; j < weights.get_cols(); j++) {
            weights.set(i, j, dis(gen));
        }
    }
    
    // Initialize bias to zero
    for(int j = 0; j < bias.get_cols(); j++) {
        bias.set(0, j, 0.0);
    }
}

Matrix FullyConnected::forward(const Matrix& input) {
    // Input size validation
    if (input.get_cols() != weights.get_rows()) {
        throw std::runtime_error("Input dimensions don't match weights");
    }

    // output = input * weights + bias
    Matrix output = input.multiply(weights);
    
    // Add bias to each row
    for(int i = 0; i < output.get_rows(); i++) {
        for(int j = 0; j < output.get_cols(); j++) {
            output.set(i, j, output.get(i, j) + bias.get(0, j));
        }
    }
    
    return output;
}

void FullyConnected::setWeights(const Matrix& new_weights) {
    if (new_weights.get_rows() != weights.get_rows() || 
        new_weights.get_cols() != weights.get_cols()) {
        throw std::runtime_error("New weights dimensions don't match");
    }
    weights = new_weights;
}

void FullyConnected::setBias(const Matrix& new_bias) {
    if (new_bias.get_rows() != 1 || 
        new_bias.get_cols() != bias.get_cols()) {
        throw std::runtime_error("New bias dimensions don't match");
    }
    bias = new_bias;
}
