#include <iostream>
#include <filesystem> // C++17 feature to check file existence
#include "DataLoader.hpp"
#include "PCA.hpp"
#include "KNN.hpp"

namespace fs = std::filesystem;

int main()
{
    try
    {
        using namespace linalg;

        PCA pca;
        KNN classifier;

        // File paths for our saved brains
        fs::path model_dir = MODELS_DIR;
        if (!fs::exists(model_dir))
        {
            fs::create_directories(model_dir);
            std::cout << "[System] Created models directory: " << model_dir << "\n";
        }
        std::string pca_model_path = (model_dir / "pca_model.bin").string();
        std::string knn_model_path = (model_dir / "knn_model.bin").string();

        // === PHASE 1: LOAD OR TRAIN ===
        if (fs::exists(pca_model_path) && fs::exists(knn_model_path))
        {
            std::cout << "\n=== FOUND SAVED MODEL ===\n";
            std::cout << "Skipping training... Loading from disk.\n";

            pca.load(pca_model_path);
            classifier.load(knn_model_path);
        }
        else
        {
            std::cout << "\n=== NO MODEL FOUND. TRAINING STARTING ===\n";

            // 1. Load Training Data
            auto train_data = DataLoader::load("data/mnist_train.csv", 20000);

            // 2. Train PCA
            int n_components = 40;
            pca.fit(train_data.images, n_components);
            pca.save(pca_model_path); // <--- SAVE

            // 3. Transform & Train KNN
            std::cout << "[Main] Transforming Training Data...\n";
            Matrix<double> train_reduced = pca.transform(train_data.images);

            classifier.fit(train_reduced, train_data.labels);
            classifier.save(knn_model_path); // <--- SAVE
        }

        // === PHASE 2: TESTING ===
        std::cout << "\n=== PHASE 2: TESTING ===\n";

        // Load Test Data
        auto test_data = DataLoader::load("data/mnist_test.csv", 1000);

        std::cout << "[Main] Transforming Test Data...\n";
        Matrix<double> test_reduced = pca.transform(test_data.images);

        // Evaluate
        double accuracy = classifier.evaluate(test_reduced, test_data.labels, 5); // k=5

        std::cout << "---------------------------\n";
        std::cout << "Model Accuracy: " << accuracy << "%\n";
        std::cout << "---------------------------\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}