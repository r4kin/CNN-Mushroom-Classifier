#include "../../include/cnn/activation.h"
#include <iostream>
#include <cassert>

void testReLUFunction() {
    std::cout << "\nTesting ReLU Function:\n";
    std::cout << "===========================\n";
    
    Matrix input(3, 3);
    input.fill(0.0);
    input.set(0, 0, -2.0);
    input.set(0, 1, -1.0);
    input.set(0, 2, 0.0);
    input.set(1, 0, 1.0);
    input.set(1, 1, 2.0);
    input.set(1, 2, -0.5);
    
    std::cout << "Input matrix:\n";
    input.print();
    
    Matrix relu_output = Activation::relu(input);
    std::cout << "\nReLU output:\n";
    relu_output.print();
}

void testReLUDerivative() {
    std::cout << "\nTesting ReLU Derivative:\n";
    std::cout << "===========================\n";
    
    Matrix input(3, 3);
    input.fill(0.0);
    input.set(0, 0, -2.0);
    input.set(0, 1, -1.0);
    input.set(1, 0, 1.0);
    input.set(1, 1, 2.0);
    
    std::cout << "Input matrix:\n";
    input.print();
    
    Matrix relu_grad = Activation::relu_derivative(input);
    std::cout << "\nReLU derivative:\n";
    relu_grad.print();
}

void testSigmoidFunction() {
    std::cout << "\nTesting Sigmoid Function:\n";
    std::cout << "===========================\n";
    
    Matrix input(3, 3);
    input.fill(0.0);
    input.set(0, 0, -2.0);
    input.set(0, 1, -1.0);
    input.set(0, 2, 0.0);
    input.set(1, 0, 1.0);
    input.set(1, 1, 2.0);
    
    std::cout << "Input matrix:\n";
    input.print();
    
    Matrix sigmoid_output = Activation::sigmoid(input);
    std::cout << "\nSigmoid output:\n";
    sigmoid_output.print();
}

void testTanhFunction() {
    std::cout << "\nTesting Tanh Function:\n";
    std::cout << "===========================\n";
    
    Matrix input(3, 3);
    input.fill(0.0);
    input.set(0, 0, -2.0);
    input.set(0, 1, -1.0);
    input.set(0, 2, 0.0);
    input.set(1, 0, 1.0);
    input.set(1, 1, 2.0);
    
    std::cout << "Input matrix:\n";
    input.print();
    
    Matrix tanh_output = Activation::tanh(input);
    std::cout << "\nTanh output:\n";
    tanh_output.print();
}

int main() {
    try {
        std::cout << "Starting Activation Tests\n";
        std::cout << "============================\n";
        
        testReLUFunction();
        testReLUDerivative();
        testSigmoidFunction();
        testTanhFunction();
        
        std::cout << "\nAll activation tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}
