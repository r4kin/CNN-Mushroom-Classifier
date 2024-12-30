#pragma once
#include "matrix.h"

class Convolution {
private:
    Matrix kernel;
    Matrix kernel_gradients; // Storage for gradients
    double learning_rate;
    int stride;
    int padding;
    
public:
    // Constructors
    Convolution(int kernel_size, double learning_rate=0.01, int stride=1, int padding=0);
    ~Convolution();
    
    // Matrix Operators
    Matrix forward(const Matrix& input);
    Matrix pad(const Matrix& input, int pad_size);
    Matrix convolve2D(const Matrix& input);
    
    // Kernel Operators
    void setKernel(const Matrix& new_kernel);
    Matrix getKernel() const;

    // Gradient Kernel Operators
    void updateKernel(const Matrix& gradients);
    Matrix getKernelGradients() const;
    
    // Activation Functions
    Matrix relu(const Matrix& input);
    
    // Utility Functions
    void printKernel() const;
    int getStride() const;
    int getPadding() const;
};