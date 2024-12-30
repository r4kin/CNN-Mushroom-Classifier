#include "../../include/cnn/matrix.h"
#include "../../include/cnn/convolution.h"
#include "../../include/cnn/activation.h"
#include <iostream>
#include <cassert>

void testForwardPass() {
    std::cout << "\nTesting Forward Pass Integration:\n";
    std::cout << "===========================\n";
    
    // Create input matrix
    Matrix input(4, 4);
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            input.set(i, j, i + j - 2.0); // Include negative values
        }
    }
    
    // Apply convolution
    Convolution conv(3, 1, 1);
    Matrix conv_output = conv.forward(input);
    
    // Apply activation
    Matrix activated = Activation::relu(conv_output);
    
    std::cout << "Original Input:\n";
    input.print();
    std::cout << "\nAfter Convolution:\n";
    conv_output.print();
    std::cout << "\nAfter ReLU:\n";
    activated.print();
}

int main() {  
    try {
        std::cout << "Starting Integration Tests\n";
        testForwardPass();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}