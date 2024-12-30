#include "../../include/cnn/activation.h"
#include <iostream>
#include <cassert>
#include <cmath>

void testReLUFunction() {
    // Keep existing implementation
}

void testReLUDerivative() {
    // Keep existing implementation
}

void testSigmoidFunction() {
    std::cout << "\nTesting Sigmoid Function:\n";
    std::cout << "===========================\n";
    
    Matrix input(3, 3);
    input.fill(0.0);
    input.set(0, 0, -2.0);  // Should be close to 0
    input.set(0, 1, -1.0);  // Should be < 0.5
    input.set(0, 2, 0.0);   // Should be 0.5
    input.set(1, 0, 1.0);   // Should be > 0.5
    input.set(1, 1, 2.0);   // Should be close to 1
    
    std::cout << "Input matrix:\n";
    input.print();
    
    Matrix sigmoid_output = Activation::sigmoid(input);
    std::cout << "\nSigmoid output:\n";
    sigmoid_output.print();
    
    Matrix sigmoid_grad = Activation::sigmoid_derivative(input);
    std::cout << "\nSigmoid derivative:\n";
    sigmoid_grad.print();
}

void testTanhFunction() {
    std::cout << "\nTesting Tanh Function:\n";
    std::cout << "===========================\n";
    
    Matrix input(3, 3);
    input.fill(0.0);
    input.set(0, 0, -2.0);  // Should be close to -1
    input.set(0, 1, -1.0);  // Should be between -1 and 0
    input.set(0, 2, 0.0);   // Should be 0
    input.set(1, 0, 1.0);   // Should be between 0 and 1
    input.set(1, 1, 2.0);   // Should be close to 1
    
    std::cout << "Input matrix:\n";
    input.print();
    
    Matrix tanh_output = Activation::tanh(input);
    std::cout << "\nTanh output:\n";
    tanh_output.print();
    
    Matrix tanh_grad = Activation::tanh_derivative(input);
    std::cout << "\nTanh derivative:\n";
    tanh_grad.print();
}

void testRangeProperties() {
    std::cout << "\nTesting Range Properties:\n";
    std::cout << "===========================\n";
    
    Matrix input(1, 3);
    input.set(0, 0, -10.0);  // Very negative
    input.set(0, 1, 0.0);    // Zero
    input.set(0, 2, 10.0);   // Very positive
    
    std::cout << "Testing extreme values:\n";
    std::cout << "Input:\n";
    input.print();
    
    std::cout << "\nReLU (should be [0, 0, 10]):\n";
    Activation::relu(input).print();
    
    std::cout << "\nSigmoid (should be [≈0, 0.5, ≈1]):\n";
    Activation::sigmoid(input).print();
    
    std::cout << "\nTanh (should be [≈-1, 0, ≈1]):\n";
    Activation::tanh(input).print();
}

int main() {
    try {
        std::cout << "Starting Activation Tests\n";
        std::cout << "============================\n";
        
        testReLUFunction();
        testReLUDerivative();
        testSigmoidFunction();
        testTanhFunction();
        testRangeProperties();  // New test
        
        std::cout << "\nAll activation tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}
