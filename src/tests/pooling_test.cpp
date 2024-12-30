#include "../../include/cnn/pooling.h"
#include <iostream>
#include <cassert>

void testMaxPooling() {
    std::cout << "\nTesting Max Pooling:\n";
    std::cout << "===========================\n";
    
    // Create test input matrix (4x4)
    Matrix input(4, 4);
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            input.set(i, j, i + j);  // Creates gradient pattern
        }
    }
    
    Pooling pool(2, 2);  // 2x2 kernel with stride 2
    std::cout << "Input matrix:\n";
    input.print();
    
    Matrix result = pool.maxPool(input);
    std::cout << "\nAfter max pooling (2x2):\n";
    result.print();
}

void testOutputDimensions() {
    std::cout << "\nTesting Output Dimensions:\n";
    std::cout << "===========================\n";
    
    Matrix input(6, 6);
    input.fill(1.0);
    
    Pooling pool(2, 2);
    Matrix result = pool.maxPool(input);
    
    std::cout << "Input dimensions: 6x6\n";
    std::cout << "Output dimensions: " << result.get_rows() << "x" << result.get_cols() << "\n";
    assert(result.get_rows() == 3 && result.get_cols() == 3);
}

int main() {
    try {
        std::cout << "Starting Pooling Tests\n";
        std::cout << "============================\n";
        
        testMaxPooling();
        testOutputDimensions();
        
        std::cout << "\nAll pooling tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}
