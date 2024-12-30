#include "../../include/cnn/matrix.h"
#include <iostream>

//Constructor Functions 

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    allocate();
}

Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
    allocate();
    
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            data[i][j]  = other.data[i][j];
        }
    }
}

Matrix::~Matrix() {
    deallocate();
}

//Memory Management Functions

void Matrix::allocate() {
    data = new double*[rows];
    for (int i = 0; i < rows; i++){
        data[i] = new double[cols];
    }
}

void Matrix::deallocate() {
    for(int i = 0; i < rows; i++){
        delete[] data[i];
    }
    delete[] data;
}

//Operation Functions

double Matrix::get(int i, int j) const {
    return data[i][j];
}

void Matrix::set(int i, int j, double value){
    data[i][j] = value;
}

Matrix Matrix::multiply(const Matrix& other) const{
    if (cols != other.rows){
        throw std::runtime_error("Matrix Multiplication Error: Dimensions don't match");
    }

    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; i++){
        for(int j = 0; j < other.cols; j++){
            double sum = 0;

            for (int k = 0; k < cols; k++){
                sum += data[i][k] * other.data[k][j];
            }

            result.data[i][j] = sum;
        }
    }

    return result;
}

Matrix Matrix::transpose() {
    Matrix result(cols, rows); // swapped rows/cols

    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = data[i][j];
        }
    }

    return result;
}

//Utility Functions

void Matrix::fill(double value) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = value;
        }
    }
}

void Matrix::print() const {
    for (int i = 0; i < rows; i++) { 
        for (int j = 0; j < cols; j++) {
            std::cout << data[i][j] << " ";
        }
        
        std::cout << std::endl;
    }
}