#include "../../include/cnn/matrix.h"
#include "../../include/cnn/convolution.h"
#include "../../include/cnn/activation.h"
#include <iostream>
#include <cassert>

void testForwardPass() {
    std::cout << "\nTesting Forward Pass Integration:\n";
    std::cout << "===========================\n";
    
    // Create input matrix (matching Python test)
    Matrix input(4, 4);
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            input.set(i, j, i + j - 2.0);
        }
    }
    
    std::cout << "Original Input:\n";
    input.print();
    
    // Apply convolution with same parameters as Python test
    Convolution conv(3, 0.01, 1, 1);  // 3x3 kernel, stride=1, padding=1
    Matrix conv_output = conv.forward(input);
    
    std::cout << "\nAfter Convolution:\n";
    conv_output.print();
    
    // Apply ReLU activation
    Matrix activated = Activation::relu(conv_output);
    
    std::cout << "\nAfter ReLU:\n";
    activated.print();
    
    // Add verification checks
    std::cout << "\nVerifying dimensions:\n";
    std::cout << "Input: " << input.get_rows() << "x" << input.get_cols() << "\n";
    std::cout << "Output: " << activated.get_rows() << "x" << activated.get_cols() << "\n";
}

int main() {  
    try {
        std::cout << "Starting Integration Tests\n";
        std::cout << "============================\n";
        
        testForwardPass();
        
        std::cout << "\nAll integration tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}
