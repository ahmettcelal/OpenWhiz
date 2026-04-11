#include <iostream>
#include "OpenWhiz/openwhiz.hpp"

int main() {
    std::cout << "=== OpenWhiz Tensor Example ===" << std::endl;

    // Create a 3x3 matrix (Rank 2 Tensor)
    ow::owTensor<float, 2>::owTensorShape shape = {{3, 3}};
    auto mat1 = ow::owTensor<float, 2>::Random(shape, 0.0f, 10.0f);
    auto mat2 = ow::owTensor<float, 2>::Ones(shape);

    std::cout << "Matrix 1 (Random):\n";
    mat1.print();

    std::cout << "\nMatrix 2 (Ones):\n";
    mat2.print();

    // Addition
    auto sum = mat1 + mat2;
    std::cout << "\nSum (Mat1 + Mat2):\n";
    sum.print();

    // Matrix Multiplication (Dot Product)
    auto dotProd = mat1.dot(mat2);
    std::cout << "\nDot Product (Mat1 . Mat2):\n";
    dotProd.print();

    return 0;
}
