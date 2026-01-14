#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <linalg/Matrix.hpp>

struct DataSet
{
    linalg::Matrix<double> images;
    linalg::Matrix<double> labels;
};

class DataLoader
{
public:
    // max_rows: 0 means "read everything"
    static DataSet load(const std::string &path, size_t max_rows = 0)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file at: " + path);
        }

        std::vector<std::vector<double>> raw_images;
        std::vector<double> raw_labels;
        std::string line;
        size_t rows_read = 0;

        std::cout << "Loading " << path << "..." << std::flush;

        while (std::getline(file, line))
        {
            if (max_rows > 0 && rows_read >= max_rows)
                break;

            std::stringstream ss(line);
            std::string val_str;
            std::vector<double> row_pixels;

            // 1. Read the Label (The first number in the row)
            if (!std::getline(ss, val_str, ','))
                continue;
            raw_labels.push_back(std::stod(val_str));

            // 2. Read the Pixels (The rest of the row)
            while (std::getline(ss, val_str, ','))
            {
                row_pixels.push_back(std::stod(val_str) / 255.0);
            }

            raw_images.push_back(row_pixels);
            rows_read++;

            if (rows_read % 2000 == 0)
                std::cout << "." << std::flush;
        }
        std::cout << " Done! (" << rows_read << " samples)\n";

        size_t n_samples = raw_images.size();
        size_t n_features = raw_images[0].size();

        linalg::Matrix<double> X(n_samples, n_features);
        linalg::Matrix<double> Y(n_samples, 1);

        for (size_t i = 0; i < n_samples; ++i)
        {
            Y(i, 0) = raw_labels[i];
            for (size_t j = 0; j < n_features; ++j)
            {
                X(i, j) = raw_images[i][j];
            }
        }

        return {X, Y};
    }
};