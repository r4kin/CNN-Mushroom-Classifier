#include "../../include/cnn/loss.h"
#include "../../include/cnn/matrix.h"
#include <cmath>

Loss::Loss(double learning_rate) : learning_rate(learning_rate) {}

double Loss::calculateLoss(const Matrix& predicted, const Matrix& target) {
    double sum_squared_error = 0.0;
    int total_elements = predicted.get_rows() * predicted.get_cols();
    
    for(int i = 0; i < predicted.get_rows(); i++) {
        for(int j = 0; j < predicted.get_cols(); j++) {
            double error = predicted.get(i, j) - target.get(i, j);
            sum_squared_error += error * error;
        }
    }
    
    return sum_squared_error / (2.0 * total_elements);  // MSE/2 for easier derivative
}

Matrix Loss::calculateGradient(const Matrix& predicted, const Matrix& target) {
    Matrix gradient(predicted.get_rows(), predicted.get_cols());
    int total_elements = predicted.get_rows() * predicted.get_cols();
    
    for(int i = 0; i < predicted.get_rows(); i++) {
        for(int j = 0; j < predicted.get_cols(); j++) {
            // Derivative of MSE with respect to predicted values
            double grad = (predicted.get(i, j) - target.get(i, j)) / total_elements;
            gradient.set(i, j, grad);
        }
    }
    
    return gradient;
}
