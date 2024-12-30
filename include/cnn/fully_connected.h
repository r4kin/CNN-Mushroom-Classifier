#pragma once
#include "matrix.h"
#include <stdexcept>

class FullyConnected {
private:
    Matrix weights;
    Matrix bias;
    
public:
    // Constructor
    FullyConnected(int input_size, int output_size);
    
    // Core operations
    Matrix forward(const Matrix& input);
    void initializeWeights();
    
    // Getters
    Matrix getWeights() const { return weights; }
    Matrix getBias() const { return bias; }
    
    // Setters
    void setWeights(const Matrix& new_weights);
    void setBias(const Matrix& new_bias);
};
