#pragma once
#include "matrix.h"

class Pooling {
private:
    int kernel_size;
    int stride;

public:
    // Constructor
    Pooling(int kernel_size = 2, int stride = 2);
    
    // Core operations
    Matrix maxPool(const Matrix& input);
    Matrix avgPool(const Matrix& input);
    
    // Utility functions
    int calculateOutputDim(int input_dim) const;
};