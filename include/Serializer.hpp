#pragma once
#include <fstream>
#include <iostream>
#include <linalg/Matrix.hpp>

namespace Serializer
{

    template <typename T>
    void save_matrix(std::ofstream &out, const linalg::Matrix<T> &mat)
    {
        size_t rows = mat.rows();
        size_t cols = mat.cols();
        out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T val = mat(i, j);
                out.write(reinterpret_cast<const char *>(&val), sizeof(T));
            }
        }
    }

    template <typename T>
    linalg::Matrix<T> load_matrix(std::ifstream &in)
    {
        size_t rows, cols;
        in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        in.read(reinterpret_cast<char *>(&cols), sizeof(cols));

        linalg::Matrix<T> mat(rows, cols);
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T val;
                in.read(reinterpret_cast<char *>(&val), sizeof(T));
                mat(i, j) = val;
            }
        }
        return mat;
    }
}