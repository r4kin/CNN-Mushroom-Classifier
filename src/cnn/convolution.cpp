#include "../../include/cnn/convolution.h"
#include <iostream>
#include <cassert>

Convolution::Convolution(int kernel_size, double learning_rate, int stride, int padding) 
    : kernel(kernel_size, kernel_size),
      kernel_gradients(kernel_size, kernel_size),
      learning_rate(learning_rate),
      stride(stride),
      padding(padding) 
{
    kernel.fill(1.0 / (kernel_size * kernel_size));
    kernel_gradients.fill(0.0);
}

Convolution::~Convolution() {
    // Destructor - Matrix class handles its own cleanup
}

Matrix Convolution::forward(const Matrix& input) {
    Matrix padded = pad(input, padding);
    return convolve2D(padded);
}

Matrix Convolution::pad(const Matrix& input, int pad_size) {
    if (pad_size == 0) return input;
    
    Matrix result(input.get_rows() + 2*pad_size, input.get_cols() + 2*pad_size);
    result.fill(0.0);
    
    for(int i = 0; i < input.get_rows(); i++) {
        for(int j = 0; j < input.get_cols(); j++) {
            result.set(i + pad_size, j + pad_size, input.get(i, j));
        }
    }
    return result;
}

Matrix Convolution::convolve2D(const Matrix& input) {
    int output_rows = (input.get_rows() - kernel.get_rows()) / stride + 1;
    int output_cols = (input.get_cols() - kernel.get_cols()) / stride + 1;
    Matrix result(output_rows, output_cols);
    
    for(int i = 0; i < output_rows; i++) {
        for(int j = 0; j < output_cols; j++) {
            double sum = 0;
            for(int k = 0; k < kernel.get_rows(); k++) {
                for(int l = 0; l < kernel.get_cols(); l++) {
                    sum += input.get(i*stride + k, j*stride + l) * kernel.get(k, l);
                }
            }
            result.set(i, j, sum);
        }
    }
    return result;
}

void Convolution::printKernel() const {
    kernel.print();
}

Matrix Convolution::relu(const Matrix& input) {
    Matrix result(input.get_rows(), input.get_cols());
    for(int i = 0; i < input.get_rows(); i++) {
        for(int j = 0; j < input.get_cols(); j++) {
            result.set(i, j, std::max(0.0, input.get(i, j)));
        }
    }
    return result;
}

void Convolution::updateKernel(const Matrix& gradients) {
    // Store the gradients
    kernel_gradients = gradients;  // Add this line
    
    // Update kernel weights
    for(int i = 0; i < kernel.get_rows(); i++) {
        for(int j = 0; j < kernel.get_cols(); j++) {
            double current = kernel.get(i, j);
            kernel.set(i, j, current - learning_rate * gradients.get(i, j));
        }
    }
}

Matrix Convolution::getKernelGradients() const {
    return kernel_gradients;
}

Matrix Convolution::getKernel() const {
    return kernel;
}

void Convolution::setKernel(const Matrix& new_kernel) {
    kernel = new_kernel;
}

int Convolution::getStride() const {
    return stride;
}

int Convolution::getPadding() const {
    return padding;
}