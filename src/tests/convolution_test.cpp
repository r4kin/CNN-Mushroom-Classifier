#include "../../include/cnn/convolution.h"
#include <iostream>
#include <cassert>

void testConvolutionOperations() {
    std::cout << "\nTesting Convolution Operations:\n";
    std::cout << "===========================\n";
    
    // Test kernel initialization
    std::cout << "Testing Kernel (3x3):\n";
    Convolution conv(3, 1, 1);  // kernel_size=3, stride=1, padding=1
    conv.printKernel();
    
    // Test padding
    std::cout << "\nTesting Padding:\n";
    Matrix input(4, 4);
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            input.set(i, j, i + j);  // Creates gradient pattern
        }
    }
    
    std::cout << "Original input:\n";
    input.print();
    
    Matrix padded = conv.pad(input, 1);
    std::cout << "Padded result:\n";
    padded.print();
    
    // Test forward convolution
    std::cout << "\nTesting Forward Convolution:\n";
    Matrix convResult = conv.forward(input);
    std::cout << "Convolution result:\n";
    convResult.print();
    
    // Test ReLU with comprehensive cases
    std::cout << "\nTesting ReLU Activation:\n";
    Matrix reluInput(3, 3);
    reluInput.fill(0.0);  // Initialize all values
    
    // Set test values to cover positive, negative, and zero cases
    reluInput.set(0, 0, -1.0);   // Negative
    reluInput.set(0, 1, 0.0);    // Zero
    reluInput.set(0, 2, 1.0);    // Positive
    reluInput.set(1, 0, -2.5);   // Large negative
    reluInput.set(1, 1, 3.7);    // Large positive
    reluInput.set(1, 2, -0.1);   // Small negative
    reluInput.set(2, 0, 0.1);    // Small positive
    reluInput.set(2, 1, -4.2);   // Large negative
    reluInput.set(2, 2, 5.5);    // Large positive
    
    std::cout << "Before ReLU:\n";
    reluInput.print();
    
    Matrix activated = conv.relu(reluInput);
    std::cout << "After ReLU:\n";
    activated.print();
}

int main() {
    try {
        std::cout << "Starting Convolution Tests\n";
        std::cout << "============================\n";
        
        testConvolutionOperations();
        
        std::cout << "\nAll convolution tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}