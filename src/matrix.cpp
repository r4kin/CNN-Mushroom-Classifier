#include <iostream>

class Matrix {
private:
    double** data;
    int rows;
    int cols;

public:
    Matrix(int rows, int cols) : rows(rows), cols(cols) {
        data = new double*[rows];
        for(int i = 0; i < rows; i++) {
            data[i] = new double[cols];
        }
    }

    ~Matrix() {
        for(int i = 0; i < rows; i++) {
            delete[] data[i];
        }
        delete[] data;
    }

    void fill(double value) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                data[i][j] = value;
            }
        }
    }

    void print() {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

int main() {
    std::cout << "Creating 3x3 matrix..." << std::endl;
    Matrix m(3, 3);
    m.fill(1.5);
    std::cout << "Matrix contents:" << std::endl;
    m.print();
    return 0;
}