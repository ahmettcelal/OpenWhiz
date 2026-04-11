#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <iomanip>
#include "OpenWhiz/openwhiz.hpp"

/**
 * @brief Synthetic data generator for PCA.
 * Creates 3D data where points mostly lie on a 2D plane: z = 0.5x + 0.8y.
 */
void generatePCAData(const std::string& filename, int count) {
    std::ofstream file(filename);
    file << "X;Y;Z;TargetDummy\n"; // 3 Features, 1 dummy target
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::normal_distribution<float> noise(0.0f, 0.5f); // Small noise on the Z axis

    for (int i = 0; i < count; ++i) {
        float x = dist(gen);
        float y = dist(gen);
        // Plane equation: z = 0.5x + 0.8y + noise
        float z = 0.5f * x + 0.8f * y + noise(gen);

        file << x << ";" << y << ";" << z << ";0\n";
    }
}

int main() {
    std::cout << "=== OpenWhiz Principal Component Analysis (PCA) Example ===\n" << std::endl;

    const std::string csvFile = "pca_data.csv";
    generatePCAData(csvFile, 500);

    // 1. Load Data
    ow::owDataset ds;
    if (!ds.loadFromCSV(csvFile)) {
        std::cerr << "Failed to load data!" << std::endl;
        return -1;
    }
    
    // We want to analyze the 3 features (X, Y, Z)
    auto data = ds.getData();
    // Exclude the last column (TargetDummy)
    ow::owTensor<float, 2> features(data.shape()[0], 3);
    for(size_t i=0; i<data.shape()[0]; ++i) {
        for(size_t j=0; j<3; ++j) features(i, j) = data(i, j);
    }

    std::cout << "Original Data Shape: " << features.shape()[0] << "x" << features.shape()[1] << std::endl;

    // 2. Initialize PCA Layer
    // Reduce from 3D to 2D
    size_t inputDim = 3;
    size_t reducedDim = 2;
    auto pcaLayer = std::make_shared<ow::owPrincipalComponentAnalysisLayer>(inputDim, reducedDim);

    // 3. Fit PCA
    std::cout << "Fitting PCA components using Power Iteration..." << std::endl;
    pcaLayer->fit(features);

    // 4. Transform Data
    auto projected = pcaLayer->forward(features);
    std::cout << "Projected Data Shape: " << projected.shape()[0] << "x" << projected.shape()[1] << std::endl;

    // 5. Inspect Principal Components (Eigenvectors)
    std::cout << "\nLearned Principal Components (Directions of max variance):" << std::endl;
    // We can see the components through XML or by creating a getter, 
    // here we use the XML export for simplicity.
    std::cout << pcaLayer->toXML() << std::endl;

    // 6. Verification: Check reconstruction error or variance preservation
    // In a plane z = 0.5x + 0.8y, the 3rd component (discarded) should be very small.
    // The first two components should capture almost all variance.
    
    float totalVar = 0;
    for(size_t i=0; i<features.size(); ++i) totalVar += features.data()[i] * features.data()[i];
    
    float projectedVar = 0;
    for(size_t i=0; i<projected.size(); ++i) projectedVar += projected.data()[i] * projected.data()[i];

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nTotal Energy (Squared Sum) Original: " << totalVar << std::endl;
    std::cout << "Total Energy (Squared Sum) Projected: " << projectedVar << std::endl;
    std::cout << "Variance Preserved: " << (projectedVar / totalVar) * 100.0f << "%" << std::endl;

    // 7. Testing a single point
    ow::owTensor<float, 2> sample(1, 3);
    sample(0, 0) = 10.0f; sample(0, 1) = 10.0f; sample(0, 2) = 13.0f; // Close to 0.5*10 + 0.8*10 = 13
    
    auto result = pcaLayer->forward(sample);
    std::cout << "\nSample 3D Point: [10, 10, 13]" << std::endl;
    std::cout << "Projected 2D Point: [" << result(0, 0) << ", " << result(0, 1) << "]" << std::endl;

    return 0;
}
