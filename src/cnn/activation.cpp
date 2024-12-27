#include "../../include/cnn/activation.h"
#include <cmath>

Matrix Activation::relu(const Matrix& input) {
    Matrix result(input.get_rows(), input.get_cols());
    for(int i = 0; i < input.get_rows(); i++) {
        for(int j = 0; j < input.get_cols(); j++) {
            result.set(i, j, std::max(0.0, input.get(i, j)));
        }
    }
    return result;
}

Matrix Activation::relu_derivative(const Matrix& input) {
    Matrix result(input.get_rows(), input.get_cols());
    for(int i = 0; i < input.get_rows(); i++) {
        for(int j = 0; j < input.get_cols(); j++) {
            result.set(i, j, input.get(i, j) > 0 ? 1.0 : 0.0);
        }
    }
    return result;
}

Matrix Activation::sigmoid(const Matrix& input) {
    Matrix result(input.get_rows(), input.get_cols());
    for(int i = 0; i < input.get_rows(); i++) {
        for(int j = 0; j < input.get_cols(); j++) {
            double x = input.get(i, j);
            result.set(i, j, 1.0 / (1.0 + std::exp(-x)));
        }
    }
    return result;
}

Matrix Activation::sigmoid_derivative(const Matrix& input) {
    Matrix sig = sigmoid(input);
    Matrix result(input.get_rows(), input.get_cols());
    for(int i = 0; i < input.get_rows(); i++) {
        for(int j = 0; j < input.get_cols(); j++) {
            double sx = sig.get(i, j);
            result.set(i, j, sx * (1.0 - sx));
        }
    }
    return result;
}

Matrix Activation::tanh(const Matrix& input) {
    Matrix result(input.get_rows(), input.get_cols());
    for(int i = 0; i < input.get_rows(); i++) {
        for(int j = 0; j < input.get_cols(); j++) {
            result.set(i, j, std::tanh(input.get(i, j)));
        }
    }
    return result;
}

Matrix Activation::tanh_derivative(const Matrix& input) {
    Matrix result(input.get_rows(), input.get_cols());
    for(int i = 0; i < input.get_rows(); i++) {
        for(int j = 0; j < input.get_cols(); j++) {
            double tx = std::tanh(input.get(i, j));
            result.set(i, j, 1.0 - tx * tx);
        }
    }
    return result;
}
