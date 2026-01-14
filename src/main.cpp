#include <iostream>
#include <fstream>
#include <linalg/Matrix.hpp>
#include "DataLoader.hpp"

using namespace linalg;

// Helper to save results for plotting
void export_to_csv(const Matrix<double> &projected, const Matrix<double> &labels, const std::string &filename)
{
    std::ofstream file(filename);
    file << "x,y,label\n";
    for (size_t i = 0; i < projected.rows(); ++i)
    {
        file << projected(i, 0) << "," << projected(i, 1) << "," << labels(i, 0) << "\n";
    }
    std::cout << "Saved plot data to " << filename << "\n";
}

int main()
{
    try
    {
        std::cout << "1. Loading Data...\n";
        DataSet data = DataLoader::load("data/mnist_test.csv", 20000);
        Matrix<double> &X = data.images;

        std::cout << "2. Centering Data...\n";
        Matrix<double> mu = X.mean();

        for (size_t i = 0; i < X.rows(); ++i)
        {
            for (size_t j = 0; j < X.cols(); ++j)
            {
                X(i, j) -= mu(0, j);
            }
        }

        std::cout << "3. Computing Covariance...\n";
        Matrix<double> cov = X.covariance();

        int num_components = 20;
        std::cout << "4. Computing Top " << num_components << " Eigenvectors...\n";
        auto eigensystem = cov.power_iteration(num_components);

        std::cout << "5. Projecting Data...\n";
        Matrix<double> V(784, num_components);

        for (int k = 0; k < num_components; ++k)
        {
            for (size_t row = 0; row < 784; ++row)
            {
                V(row, k) = eigensystem.eigenvectors[k](row, 0);
            }
        }

        Matrix<double> X_reduced = X * V;

        std::cout << "\n--- PCA Complete ---\n";
        std::cout << "Original Size: " << X.rows() << " x " << X.cols() << "\n";
        std::cout << "Reduced Size:  " << X_reduced.rows() << " x " << X_reduced.cols() << "\n";

        std::cout << "\nFirst 5 Samples (2D coords):\n";
        for (int i = 0; i < 5; ++i)
        {
            std::cout << "Label " << data.labels(i, 0) << ": ("
                      << X_reduced(i, 0) << ", " << X_reduced(i, 1) << ")\n";
        }

        export_to_csv(X_reduced, data.labels, "pca_viz.csv");
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}