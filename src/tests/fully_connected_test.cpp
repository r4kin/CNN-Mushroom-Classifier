#include "../../include/cnn/fully_connected.h"
#include <iostream>
#include <cassert>

void testWeightInitialization() {
    std::cout << "\nTesting Weight Initialization:\n";
    std::cout << "===========================\n";
    
    FullyConnected fc(3, 2);  // 3 inputs, 2 outputs
    
    std::cout << "Initial weights:\n";
    fc.getWeights().print();
    
    std::cout << "Initial bias:\n";
    fc.getBias().print();
}

void testForwardPass() {
    std::cout << "\nTesting Forward Pass:\n";
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

void testWeightModification() {
    std::cout << "\nTesting Weight Modification:\n";
    std::cout << "===========================\n";
    
    FullyConnected fc(2, 2);
    
    // Test weight modification
    Matrix new_weights(2, 2);
    new_weights.fill(0.5);
    fc.setWeights(new_weights);
    
    Matrix new_bias(1, 2);
    new_bias.fill(0.1);
    fc.setBias(new_bias);
    
    std::cout << "Modified weights:\n";
    fc.getWeights().print();
    
    std::cout << "\nModified bias:\n";
    fc.getBias().print();
}

void testDimensionality() {
    std::cout << "\nTesting Dimensionality:\n";
    std::cout << "===========================\n";
    
    FullyConnected fc(3, 2);
    
    // Test batch processing
    Matrix batch_input(4, 3);  // 4 samples, 3 features each
    batch_input.fill(1.0);
    
    Matrix output = fc.forward(batch_input);
    std::cout << "Input dimensions: " << batch_input.get_rows() << "x" << batch_input.get_cols() << "\n";
    std::cout << "Output dimensions: " << output.get_rows() << "x" << output.get_cols() << "\n";
    
    // Verify dimensions
    assert(output.get_rows() == 4);
    assert(output.get_cols() == 2);
}

int main() {
    try {
        std::cout << "Starting Fully Connected Layer Tests\n";
        std::cout << "============================\n";
        
        testWeightInitialization();
        testForwardPass();
        testWeightModification();
        testDimensionality();
        
        std::cout << "\nAll fully connected layer tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}
