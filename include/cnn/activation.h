#pragma once
#include "matrix.h"

class Activation {
public:
    static Matrix relu(const Matrix& input);
    static Matrix relu_derivative(const Matrix& input);
    static Matrix sigmoid(const Matrix& input);
    static Matrix sigmoid_derivative(const Matrix& input);
    static Matrix tanh(const Matrix& input);
    static Matrix tanh_derivative(const Matrix& input);
};