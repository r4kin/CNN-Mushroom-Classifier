#pragma once

class Matrix {
private:
    double** data;
    int rows;
    int cols;

public:
    //Constructors
    Matrix(int rows, int cols);
    Matrix(const Matrix& other);
    ~Matrix();

    //Memory management
    void allocate();
    void deallocate();

    //Operations
    double get(int i, int j) const;
    void set(int i, int j, double value);
    Matrix multiply(const Matrix& other);
    Matrix transpose();

    //Utility
    void fill(double value);
    void print() const;

    int get_rows() const { 
        return rows; 
    }
    int get_cols() const { 
        return cols; 
    }
};