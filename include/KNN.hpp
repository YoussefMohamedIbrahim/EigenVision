#pragma once
#include <linalg/Matrix.hpp>
#include <limits>
#include <cmath>
#include <vector>
#include <map>
#include <algorithm>
#include "Serializer.hpp"

class KNN
{
public:
    KNN() : train_features_(0, 0), train_labels_(0, 0) {}

    void fit(const linalg::Matrix<double> &features, const linalg::Matrix<double> &labels)
    {
        train_features_ = features;
        train_labels_ = labels;
    }

    double predict(const linalg::Matrix<double> &query_vector, int k = 5) const
    {
        std::vector<std::pair<double, double>> neighbors;
        neighbors.reserve(train_features_.rows());

        for (size_t i = 0; i < train_features_.rows(); ++i)
        {
            double dist = 0.0;
            for (size_t j = 0; j < train_features_.cols(); ++j)
            {
                double diff = query_vector(0, j) - train_features_(i, j);
                dist += diff * diff;
            }
            neighbors.push_back({dist, train_labels_(i, 0)});
        }

        std::partial_sort(neighbors.begin(), neighbors.begin() + k, neighbors.end());

        std::map<double, int> votes;
        for (int i = 0; i < k; ++i)
        {
            votes[neighbors[i].second]++;
        }

        double best_label = -1;
        int max_votes = -1;
        for (auto const &[label, count] : votes)
        {
            if (count > max_votes)
            {
                max_votes = count;
                best_label = label;
            }
        }
        return best_label;
    }

    double evaluate(const linalg::Matrix<double> &test_features, const linalg::Matrix<double> &test_labels, int k = 5) const
    {
        int correct = 0;
        int total = test_features.rows();
        std::cout << "[KNN] Evaluating " << total << " samples (k=" << k << ")...\n";

        for (int i = 0; i < total; ++i)
        {
            linalg::Matrix<double> row_vec(1, test_features.cols());
            for (size_t c = 0; c < test_features.cols(); ++c)
                row_vec(0, c) = test_features(i, c);

            if (std::abs(predict(row_vec, k) - test_labels(i, 0)) < 0.1)
                correct++;
            if ((i + 1) % 100 == 0)
                std::cout << "." << std::flush;
        }
        std::cout << "\n";
        return (double)correct / total * 100.0;
    }

    void save(const std::string &filepath) const
    {
        std::ofstream out(filepath, std::ios::binary);
        if (!out)
            throw std::runtime_error("Cannot save KNN model to " + filepath);

        Serializer::save_matrix(out, train_features_);
        Serializer::save_matrix(out, train_labels_);
        std::cout << "[KNN] Saved model to " << filepath << "\n";
    }

    void load(const std::string &filepath)
    {
        std::ifstream in(filepath, std::ios::binary);
        if (!in)
            throw std::runtime_error("Cannot load KNN model from " + filepath);

        train_features_ = Serializer::load_matrix<double>(in);
        train_labels_ = Serializer::load_matrix<double>(in);
        std::cout << "[KNN] Loaded model from " << filepath << "\n";
    }

private:
    linalg::Matrix<double> train_features_;
    linalg::Matrix<double> train_labels_;
};