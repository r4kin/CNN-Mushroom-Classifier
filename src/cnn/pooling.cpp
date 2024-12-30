#include "../../include/cnn/pooling.h"
#include <algorithm>
#include <limits>  

Pooling::Pooling(int kernel_size, int stride) 
    : kernel_size(kernel_size), stride(stride) {}

Matrix Pooling::maxPool(const Matrix& input) {
    int output_rows = calculateOutputDim(input.get_rows());
    int output_cols = calculateOutputDim(input.get_cols());
    Matrix output(output_rows, output_cols);
    
    for(int i = 0; i < output_rows; i++) {
        for(int j = 0; j < output_cols; j++) {
            double max_val = -std::numeric_limits<double>::infinity();
            
            // Process each kernel window
            for(int k = 0; k < kernel_size; k++) {
                for(int l = 0; l < kernel_size; l++) {
                    int input_i = i * stride + k;
                    int input_j = j * stride + l;
                    max_val = std::max(max_val, input.get(input_i, input_j));
                }
            }
            output.set(i, j, max_val);
        }
    }
    return output;
}

int Pooling::calculateOutputDim(int input_dim) const {
    return (input_dim - kernel_size) / stride + 1;
}
