#include "../../include/cnn/backpropagation.h"
#include <iostream>
#include <cassert>

void testBackpropFC() {
    std::cout << "\nTesting FC Backpropagation:\n";
    std::cout << "===========================\n";
    
    // Create test data
    Matrix input(2, 3);  // 2 samples, 3 features
    input.set(0, 0, 1.0); input.set(0, 1, 2.0); input.set(0, 2, 3.0);
    input.set(1, 0, 0.5); input.set(1, 1, 1.5); input.set(1, 2, 2.5);
    
    FullyConnected fc(3, 2);  // 3 inputs, 2 outputs
    
    // Create gradient
    Matrix gradient(2, 2);  // 2 samples, 2 outputs
    gradient.set(0, 0, 0.1); gradient.set(0, 1, -0.1);
    gradient.set(1, 0, 0.2); gradient.set(1, 1, -0.2);
    
    std::cout << "Input:\n";
    input.print();
    
    std::cout << "\nInitial weights:\n";
    fc.getWeights().print();
    
    std::cout << "\nInitial bias:\n";
    fc.getBias().print();
    
    std::cout << "\nGradient:\n";
    gradient.print();
    
    Backpropagation backprop(0.01);
    Matrix input_gradients = backprop.backpropFC(gradient, input, fc);
    
    std::cout << "\nInput Gradients:\n";
    input_gradients.print();
    
    std::cout << "\nUpdated weights:\n";
    fc.getWeights().print();
    
    std::cout << "\nUpdated bias:\n";
    fc.getBias().print();
}

void testBackpropPool() {
    std::cout << "\nTesting Pool Backpropagation:\n";
    std::cout << "===========================\n";
    
    // Create 4x4 input
    Matrix input(4, 4);
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            input.set(i, j, i * 4 + j);
        }
    }
    
    Pooling pool(2, 2);  // 2x2 kernel, stride 2
    
    // Create 2x2 gradient (since 4x4 input with 2x2 pooling gives 2x2 output)
    Matrix gradient(2, 2);
    gradient.fill(1.0);
    
    std::cout << "Input:\n";
    input.print();
    
    std::cout << "\nGradient:\n";
    gradient.print();
    
    Backpropagation backprop(0.01);
    Matrix input_gradients = backprop.backpropPool(gradient, input, pool);
    
    std::cout << "\nInput Gradients:\n";
    input_gradients.print();
}

void testBackpropConv() {
    std::cout << "\nTesting Conv Backpropagation:\n";
    std::cout << "===========================\n";
    
    // Create 4x4 input
    Matrix input(4, 4);
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            input.set(i, j, i + j);
        }
    }
    
    Convolution conv(3, 0.01, 1, 1);  // 3x3 kernel, learning_rate=0.01, stride=1, padding=1
    
    // Create gradient matching convolution output size
    Matrix gradient(4, 4);
    gradient.fill(0.1);
    
    std::cout << "Input:\n";
    input.print();
    
    std::cout << "\nInitial kernel:\n";
    conv.getKernel().print();
    
    std::cout << "\nGradient:\n";
    gradient.print();
    
    Backpropagation backprop(0.01);
    Matrix input_gradients = backprop.backpropConv(gradient, input, conv);
    
    std::cout << "\nInput Gradients:\n";
    input_gradients.print();
    
    std::cout << "\nUpdated kernel:\n";
    conv.getKernel().print();
}

void testDimensionValidation() {
    std::cout << "\nTesting Dimension Validation:\n";
    std::cout << "===========================\n";
    
    try {
        // Test FC validation
        Matrix input(2, 3);
        Matrix gradient(2, 4);  // Wrong output dimension
        FullyConnected fc(3, 2);
        Backpropagation backprop;
        
        backprop.backpropFC(gradient, input, fc);
        std::cout << "FC validation failed to catch dimension mismatch\n";
    } catch (const std::runtime_error& e) {
        std::cout << "Successfully caught FC dimension mismatch: " << e.what() << std::endl;
    }
}

int main() {
    try {
        std::cout << "Starting Backpropagation Tests\n";
        std::cout << "============================\n";
        
        testBackpropFC();
        testBackpropPool();
        testBackpropConv();
        testDimensionValidation();
        
        std::cout << "\nAll backpropagation tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}
