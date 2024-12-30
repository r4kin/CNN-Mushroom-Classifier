#include "../../include/cnn/loss.h"
#include <iostream>
#include <cassert>
#include <cmath>

void testLossCalculation() {
    std::cout << "\nTesting Loss Calculation:\n";
    std::cout << "===========================\n";
    
    // Create predicted and target matrices
    Matrix predicted(2, 2);
    Matrix target(2, 2);
    
    // Test Case 1: Perfect prediction
    std::cout << "Test Case 1 - Perfect Prediction:\n";
    predicted.fill(1.0);
    target.fill(1.0);
    
    Loss loss(0.01);
    double loss_value = loss.calculateLoss(predicted, target);
    
    std::cout << "Predicted values:\n";
    predicted.print();
    std::cout << "Target values:\n";
    target.print();
    std::cout << "MSE Loss (should be 0): " << loss_value << "\n\n";
    
    // Test Case 2: Complete mismatch
    std::cout << "Test Case 2 - Complete Mismatch:\n";
    predicted.fill(1.0);
    target.fill(0.0);
    
    loss_value = loss.calculateLoss(predicted, target);
    Matrix gradients = loss.calculateGradient(predicted, target);
    
    std::cout << "Predicted values:\n";
    predicted.print();
    std::cout << "Target values:\n";
    target.print();
    std::cout << "MSE Loss: " << loss_value << "\n";
    std::cout << "Gradients:\n";
    gradients.print();
    
    // Test Case 3: Mixed values
    std::cout << "\nTest Case 3 - Mixed Values:\n";
    predicted.set(0, 0, 0.5);
    predicted.set(0, 1, 0.8);
    predicted.set(1, 0, 0.1);
    predicted.set(1, 1, 0.3);
    
    target.set(0, 0, 1.0);
    target.set(0, 1, 1.0);
    target.set(1, 0, 0.0);
    target.set(1, 1, 0.0);
    
    loss_value = loss.calculateLoss(predicted, target);
    gradients = loss.calculateGradient(predicted, target);
    
    std::cout << "Predicted values:\n";
    predicted.print();
    std::cout << "Target values:\n";
    target.print();
    std::cout << "MSE Loss: " << loss_value << "\n";
    std::cout << "Gradients:\n";
    gradients.print();
}

void testLearningRate() {
    std::cout << "\nTesting Learning Rate:\n";
    std::cout << "===========================\n";
    
    Loss loss(0.01);
    std::cout << "Initial learning rate: " << loss.getLearningRate() << "\n";
    
    loss.setLearningRate(0.001);
    std::cout << "Updated learning rate: " << loss.getLearningRate() << "\n";
}

int main() {
    try {
        std::cout << "Starting Loss Function Tests\n";
        std::cout << "============================\n";
        
        testLossCalculation();
        testLearningRate();
        
        std::cout << "\nAll loss function tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}
