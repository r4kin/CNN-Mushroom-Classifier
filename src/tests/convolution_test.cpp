#include "../../include/cnn/convolution.h"
#include <iostream>
#include <cassert>

// New test function for constructor and getters
void testConstructorAndGetters() {
    std::cout << "\nTesting Constructor and Getters:\n";
    std::cout << "===========================\n";
    
    Convolution conv(3, 0.01, 1, 1);  // kernel_size=3, learning_rate=0.01, stride=1, padding=1
    
    // Test getters
    std::cout << "Stride: " << conv.getStride() << " (expected: 1)" << std::endl;
    std::cout << "Padding: " << conv.getPadding() << " (expected: 1)" << std::endl;
    
    // Test kernel getter and setter
    Matrix original_kernel = conv.getKernel();
    std::cout << "Original kernel:\n";
    original_kernel.print();
    
    // Test kernel setter
    Matrix new_kernel(3, 3);
    new_kernel.fill(0.5);
    conv.setKernel(new_kernel);
    std::cout << "\nAfter setting new kernel:\n";
    conv.printKernel();
}

void testConvolutionOperations() {
    std::cout << "\nTesting Convolution Operations:\n";
    std::cout << "===========================\n";
    
    // Test kernel initialization
    std::cout << "Testing Kernel (3x3):\n";
    Convolution conv(3, 0.01, 1, 1);  // Added learning_rate parameter
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
    reluInput.fill(0.0);
    
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


// New test function for gradient-related operations
void testGradientOperations() {
    std::cout << "\nTesting Gradient Operations:\n";
    std::cout << "===========================\n";
    
    Convolution conv(2, 0.1);  // 2x2 kernel, learning_rate=0.1
    
    // Test kernel gradients
    Matrix gradients(2, 2);
    gradients.fill(0.5);
    
    std::cout << "Original kernel:\n";
    conv.printKernel();
    
    conv.updateKernel(gradients);
    std::cout << "\nKernel after gradient update:\n";
    conv.printKernel();
    
    Matrix stored_gradients = conv.getKernelGradients();
    std::cout << "\nStored gradients:\n";
    stored_gradients.print();
}

int main() {
    try {
        std::cout << "Starting Convolution Tests\n";
        std::cout << "============================\n";
        
        testConstructorAndGetters();    // New test
        testConvolutionOperations();    // Existing test
        testGradientOperations();       // New test
        
        std::cout << "\nAll convolution tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}
