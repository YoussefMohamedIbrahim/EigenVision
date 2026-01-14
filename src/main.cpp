#include <iostream>
#include <linalg/Matrix.hpp>

int main()
{
    linalg::Matrix<double> A(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 1;
    A(1, 0) = 0;
    A(1, 1) = 2;

    A.print();
    A.inverse().print();
    A.transpose().print();
    (A * A.inverse()).print();

    return 0;
}