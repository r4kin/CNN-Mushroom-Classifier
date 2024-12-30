#pragma once
#include "matrix.h"

class Loss {
private:
    double learning_rate;
    
public:
    Loss(double learning_rate = 0.01);
    
    // Forward pass
    double calculateLoss(const Matrix& predicted, const Matrix& target);
    
    // Backward pass
    Matrix calculateGradient(const Matrix& predicted, const Matrix& target);
    
    // Getters/Setters
    double getLearningRate() const { return learning_rate; }
    void setLearningRate(double rate) { learning_rate = rate; }
};
