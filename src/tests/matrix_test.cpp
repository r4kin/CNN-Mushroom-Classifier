#include "../../include/cnn/matrix.h"
#include <iostream>
#include <cassert>

// Test memory management and constructors
void testMemoryManagement() {
    std::cout << "\nTesting Memory Management:\n";
    {
        Matrix m1(3, 3);
        m1.fill(1.0);
        std::cout << "Original matrix:\n";
        m1.print();
        
        // Test deep copy
        Matrix m2 = m1;  // Copy constructor
        m2.set(0, 0, 5.0);
        std::cout << "\nAfter modifying copy (original should be unchanged):\n";
        m1.print();
    } // Destructor called here
}

// Test basic operations
void testOperations() {
    std::cout << "\nTesting Matrix Operations:\n";
    
    Matrix m1(2, 2);
    m1.fill(2.0);
    std::cout << "Matrix 1:\n";
    m1.print();

    // Test multiplication
    Matrix m2(2, 2);
    m2.fill(3.0);
    std::cout << "\nMatrix 2:\n";
    m2.print();
    
    Matrix result = m1.multiply(m2);
    std::cout << "\nMultiplication result:\n";
    result.print();

    // Test transpose
    Matrix transposed = m1.transpose();
    std::cout << "\nTransposed Matrix 1:\n";
    transposed.print();
}

// Test error handling
void testErrorHandling() {
    std::cout << "\nTesting Error Handling:\n";
    try {
        Matrix m1(2, 2);
        Matrix m2(3, 2);
        Matrix result = m1.multiply(m2);  // Should throw error
    } catch (const std::runtime_error& e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
    }
}

void testAssignmentOperator() {
    std::cout << "\nTesting Assignment Operator:\n";
    Matrix m1(2, 2);
    m1.fill(1.0);
    std::cout << "Original Matrix m1:\n";
    m1.print();

    Matrix m2(2, 2);
    m2.fill(2.0);
    std::cout << "\nOriginal Matrix m2:\n";
    m2.print();

    m2 = m1;  // Test assignment
    std::cout << "\nAfter m2 = m1:\n";
    m2.print();

    // Modify m2 to verify deep copy
    m2.set(0, 0, 5.0);
    std::cout << "\nAfter modifying m2 (m1 should be unchanged):\n";
    m1.print();
}

// Function for add and subtract
void testAddSubtract() {
    std::cout << "\nTesting Addition and Subtraction:\n";
    Matrix m1(2, 2);
    m1.fill(3.0);
    std::cout << "Matrix m1:\n";
    m1.print();

    Matrix m2(2, 2);
    m2.fill(1.0);
    std::cout << "\nMatrix m2:\n";
    m2.print();

    Matrix sum = m1.add(m2);
    std::cout << "\nSum (m1 + m2):\n";
    sum.print();

    Matrix diff = m1.subtract(m2);
    std::cout << "\nDifference (m1 - m2):\n";
    diff.print();

    // Test error handling for mismatched dimensions
    try {
        Matrix m3(2, 3);
        Matrix result = m1.add(m3);  // Should throw error
    } catch (const std::runtime_error& e) {
        std::cout << "\nCaught expected error for addition: " << e.what() << std::endl;
    }
}

int main() {
    try {
        std::cout << "Starting Matrix Class Tests\n";
        std::cout << "===========================\n";
        
        testMemoryManagement();
        testOperations();
        testErrorHandling();
        testAssignmentOperator();  // New test
        testAddSubtract();         // New test
        
        std::cout << "\nAll tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}