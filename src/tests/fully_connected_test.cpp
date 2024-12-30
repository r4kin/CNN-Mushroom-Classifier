#include "../../include/cnn/fully_connected.h"
#include <iostream>
#include <cassert>

void testFullyConnectedLayer() {
    std::cout << "\nTesting Fully Connected Layer:\n";
    std::cout << "===========================\n";
    
    // Create input
    Matrix input(1, 3);  // 1x3 input
    input.set(0, 0, 1.0);
    input.set(0, 1, 2.0);
    input.set(0, 2, 3.0);
    
    // Create layer
    FullyConnected fc(3, 2);  // 3 inputs, 2 outputs
    
    std::cout << "Input:\n";
    input.print();
    
    Matrix output = fc.forward(input);
    std::cout << "\nOutput:\n";
    output.print();
    
    // Verify output dimensions
    assert(output.get_rows() == 1);
    assert(output.get_cols() == 2);
}

int main() {
    try {
        std::cout << "Starting Fully Connected Layer Tests\n";
        std::cout << "============================\n";
        
        testFullyConnectedLayer();
        
        std::cout << "\nAll fully connected layer tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}
