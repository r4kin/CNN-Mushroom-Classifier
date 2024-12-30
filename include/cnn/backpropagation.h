#pragma once
#include "matrix.h"
#include "fully_connected.h"
#include "convolution.h"
#include "pooling.h"
#include <stdexcept>

class Backpropagation {
private:
    double learning_rate;
    
    // Validation methods
    void validateFCDimensions(const Matrix& gradient, const Matrix& input, const FullyConnected& fc_layer) const;
    void validateConvDimensions(const Matrix& gradient, const Matrix& input, const Convolution& conv_layer) const;
    void validatePoolDimensions(const Matrix& gradient, const Matrix& input, const Pooling& pool_layer) const;
    
public:
    // Constructor
    Backpropagation(double learning_rate = 0.01);
    
    // Backward passes for each layer type
    Matrix backpropFC(const Matrix& gradient, const Matrix& input, FullyConnected& fc_layer);
    Matrix backpropConv(const Matrix& gradient, const Matrix& input, Convolution& conv_layer);
    Matrix backpropPool(const Matrix& gradient, const Matrix& input, Pooling& pool_layer);
    
    // Getters/Setters
    double getLearningRate() const { return learning_rate; }
    void setLearningRate(double rate) { learning_rate = rate; }
};
