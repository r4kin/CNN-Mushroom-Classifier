#include "../../include/cnn/backpropagation.h"
#include <random>
#include <limits>   

Backpropagation::Backpropagation(double learning_rate) : learning_rate(learning_rate) {}

// Validation method implementations
void Backpropagation::validateFCDimensions(const Matrix& gradient, const Matrix& input, const FullyConnected& fc_layer) const {
    if (gradient.get_cols() != fc_layer.getWeights().get_cols()) {
        throw std::runtime_error("Gradient dimensions don't match FC layer output size");
    }
    if (input.get_cols() != fc_layer.getWeights().get_rows()) {
        throw std::runtime_error("Input dimensions don't match FC layer input size");
    }
}

void Backpropagation::validateConvDimensions(const Matrix& gradient, const Matrix& input, const Convolution& conv_layer) const {
    int expected_output_rows = (input.get_rows() - conv_layer.getKernel().get_rows() + 2 * conv_layer.getPadding()) / conv_layer.getStride() + 1;
    int expected_output_cols = (input.get_cols() - conv_layer.getKernel().get_cols() + 2 * conv_layer.getPadding()) / conv_layer.getStride() + 1;
    
    if (gradient.get_rows() != expected_output_rows || gradient.get_cols() != expected_output_cols) {
        throw std::runtime_error("Gradient dimensions don't match convolution output dimensions");
    }
}

void Backpropagation::validatePoolDimensions(const Matrix& gradient, const Matrix& input, const Pooling& pool_layer) const {
    int expected_output_rows = (input.get_rows() - pool_layer.getKernelSize()) / pool_layer.getStride() + 1;
    int expected_output_cols = (input.get_cols() - pool_layer.getKernelSize()) / pool_layer.getStride() + 1;
    
    if (gradient.get_rows() != expected_output_rows || gradient.get_cols() != expected_output_cols) {
        throw std::runtime_error("Gradient dimensions don't match pooling output dimensions");
    }
}

Matrix Backpropagation::backpropFC(const Matrix& gradient, const Matrix& input, FullyConnected& fc_layer) {
    // Validate dimensions
    validateFCDimensions(gradient, input, fc_layer);
    
    // Calculate weight gradients
    Matrix weight_gradients = input.transpose().multiply(gradient);
    
    // Calculate input gradients for the previous layer
    Matrix input_gradients = gradient.multiply(fc_layer.getWeights().transpose());
    
    // Update weights and biases
    Matrix weights = fc_layer.getWeights();
    Matrix bias = fc_layer.getBias();
    
    // Apply gradients with learning rate
    for(int i = 0; i < weights.get_rows(); i++) {
        for(int j = 0; j < weights.get_cols(); j++) {
            double update = learning_rate * weight_gradients.get(i, j);
            weights.set(i, j, weights.get(i, j) - update);
        }
    }
    
    // Update bias
    for(int j = 0; j < bias.get_cols(); j++) {
        double bias_gradient = 0.0;
        for(int i = 0; i < gradient.get_rows(); i++) {
            bias_gradient += gradient.get(i, j);
        }
        bias.set(0, j, bias.get(0, j) - learning_rate * bias_gradient);
    }
    
    // Update the layer's weights and bias
    fc_layer.setWeights(weights);
    fc_layer.setBias(bias);
    
    return input_gradients;
}

Matrix Backpropagation::backpropConv(const Matrix& gradient, const Matrix& input, Convolution& conv_layer) {
    // Validate dimensions
    validateConvDimensions(gradient, input, conv_layer);
    
    // For simplicity, we'll implement a basic version
    // In a full implementation, this would involve:
    // 1. Calculating kernel gradients
    // 2. Updating kernel weights
    // 3. Computing input gradients
    return gradient; // Placeholder for now
}

Matrix Backpropagation::backpropPool(const Matrix& gradient, const Matrix& input, Pooling& pool_layer) {
    // Validate dimensions
    validatePoolDimensions(gradient, input, pool_layer);
    
    // For max pooling, gradients flow only through the maximum elements
    Matrix input_gradients(input.get_rows(), input.get_cols());
    input_gradients.fill(0.0);
    
    int kernel_size = pool_layer.getKernelSize();
    int stride = pool_layer.getStride();
    
    for(int i = 0; i < gradient.get_rows(); i++) {
        for(int j = 0; j < gradient.get_cols(); j++) {
            // Find max element position in the corresponding input region
            double max_val = -std::numeric_limits<double>::max();
            int max_i = 0, max_j = 0;
            
            for(int k = 0; k < kernel_size; k++) {
                for(int l = 0; l < kernel_size; l++) {
                    int input_i = i * stride + k;
                    int input_j = j * stride + l;
                    if(input.get(input_i, input_j) > max_val) {
                        max_val = input.get(input_i, input_j);
                        max_i = input_i;
                        max_j = input_j;
                    }
                }
            }
            
            // Assign gradient to max element position
            input_gradients.set(max_i, max_j, gradient.get(i, j));
        }
    }
    
    return input_gradients;
}
