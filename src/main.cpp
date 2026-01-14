#include <iostream>
#include <linalg/Matrix.hpp>

int main()
{
    auto matrix_1 = linalg::Matrix<int>(3, 3, 1);
    auto matrix_2 = linalg::Matrix<int>(3, 3, 1);
    (matrix_1 * matrix_2).print();
    return 0;
}