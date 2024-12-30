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
