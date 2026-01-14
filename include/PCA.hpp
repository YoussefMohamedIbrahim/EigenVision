#pragma once
#include <linalg/Matrix.hpp>
#include <iostream>
#include <string>
#include "Serializer.hpp"

class PCA
{
public:
    PCA() : mean_(0, 0), components_(0, 0) {}

    void fit(const linalg::Matrix<double> &X, int n_components)
    {
        std::cout << "[PCA] Computing Mean...\n";
        mean_ = X.mean();

        std::cout << "[PCA] Centering Data...\n";
        linalg::Matrix<double> centered = X;
        for (size_t i = 0; i < centered.rows(); ++i)
        {
            for (size_t j = 0; j < centered.cols(); ++j)
            {
                centered(i, j) -= mean_(0, j);
            }
        }

        std::cout << "[PCA] Computing Covariance...\n";
        linalg::Matrix<double> cov = centered.covariance();

        std::cout << "[PCA] Power Iteration (" << n_components << " components)...\n";
        auto eigensystem = cov.power_iteration(n_components);

        components_ = linalg::Matrix<double>(X.cols(), n_components);
        for (int k = 0; k < n_components; ++k)
        {
            for (size_t row = 0; row < X.cols(); ++row)
            {
                components_(row, k) = eigensystem.eigenvectors[k](row, 0);
            }
        }
        std::cout << "[PCA] Fit Complete.\n";
    }

    linalg::Matrix<double> transform(const linalg::Matrix<double> &X) const
    {
        linalg::Matrix<double> centered = X;
        for (size_t i = 0; i < centered.rows(); ++i)
        {
            for (size_t j = 0; j < centered.cols(); ++j)
            {
                centered(i, j) -= mean_(0, j);
            }
        }
        return centered * components_;
    }

    void save(const std::string &filepath) const
    {
        std::ofstream out(filepath, std::ios::binary);
        if (!out)
            throw std::runtime_error("Cannot save PCA model to " + filepath);

        Serializer::save_matrix(out, mean_);
        Serializer::save_matrix(out, components_);
        std::cout << "[PCA] Saved model to " << filepath << "\n";
    }

    void load(const std::string &filepath)
    {
        std::ifstream in(filepath, std::ios::binary);
        if (!in)
            throw std::runtime_error("Cannot load PCA model from " + filepath);

        mean_ = Serializer::load_matrix<double>(in);
        components_ = Serializer::load_matrix<double>(in);
        std::cout << "[PCA] Loaded model from " << filepath << "\n";
    }

private:
    linalg::Matrix<double> mean_;
    linalg::Matrix<double> components_;
};