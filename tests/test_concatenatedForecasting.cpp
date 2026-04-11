#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <memory>
#include <cassert>
#include "OpenWhiz/owNeuralNetwork.hpp"

#ifndef OW_PI
#define OW_PI 3.1415926535f
#endif

// Helper to generate synthetic forecasting data
void generateSyntheticData(const std::string& filename, int points) {
    std::ofstream file(filename);
    file << "Sales;Temperature;Hour;DayOfWeek;Month;DayOfMonth\n";
    for (int i = 0; i < points; ++i) {
        float hour = (float)(i % 24);
        float dow = (float)((i / 24) % 7);
        float month = (float)(((i / (24 * 30)) % 12) + 1);
        float dom = (float)((i / 24) % 31 + 1);
        
        float sales = 100.0f + 20.0f * std::sin(2.0f * OW_PI * hour / 24.0f) 
                             + 10.0f * std::sin(2.0f * OW_PI * dow / 7.0f)
                             + (rand() % 10);
        
        float temp = 20.0f + 5.0f * std::sin(2.0f * OW_PI * hour / 24.0f) + (i * 0.01f);

        file << sales << ";" << temp << ";" << hour << ";" << dow << ";" << month << ";" << dom << "\n";
    }
    file.close();
}

int main() {
    std::cout << "=== Running Test: Concatenated Forecasting ===" << std::endl;

    // 1. Generate Synthetic Data
    std::string csv_path = "tests/test_concatenated_data.csv";
    generateSyntheticData(csv_path, 200);
    std::cout << "Generated synthetic data." << std::endl;

    // 2. Load and Prepare Dataset
    auto dataset = std::make_shared<ow::owDataset>();
    dataset->setDelimiter(';');
    if (!dataset->loadFromCSV(csv_path)) {
        std::cerr << "Failed to load CSV." << std::endl;
        return -1;
    }

    int windowSize = 5; // Smaller window for faster test
    int horizon = 1;
    dataset->prepareForecastingData(windowSize, horizon, 0, 1);
    
    int inputDim = dataset->getInputVariableNum();
    std::cout << "Data prepared. Input Dimension: " << inputDim << std::endl;

    // 3. Setup Neural Network
    ow::owNeuralNetwork nn;
    nn.setDataset(dataset);

    nn.getOptimizer()->setGradientClipThreshold(1.0f);
    nn.getOptimizer()->setLearningRate(0.01f);

    auto normLayer = std::make_shared<ow::owNormalizationLayer>((size_t)inputDim);
    nn.addLayer(normLayer);

    auto anomalyLayer = std::make_shared<ow::owAnomalyDetectionLayer>(3.0f);
    nn.addLayer(anomalyLayer);

    nn.createNeuralNetwork({32, 16}, "ReLU", "Identity");

    // 4. Train for a few epochs
    nn.setMaximumEpochNum(10);
    std::cout << "Starting training (10 epochs)..." << std::endl;
    nn.train();

    // 5. Verify prediction
    auto trainInput = dataset->getTrainInput();
    size_t lastIdx = trainInput.shape()[0] - 1;
    ow::owTensor<float, 2> sampleInput({1, (size_t)inputDim});
    for(size_t i=0; i<(size_t)inputDim; ++i) sampleInput(0, i) = trainInput(lastIdx, i);

    auto prediction = nn.forward(sampleInput);

    float result = prediction(0, 0);
    std::cout << "Predicted value: " << result << std::endl;

    // Check for NaN
    if (std::isnan(result)) {
        std::cerr << "Test FAILED: Prediction is NaN!" << std::endl;
        return -1;
    }

    std::cout << "Test PASSED: Prediction is valid (not NaN)." << std::endl;

    return 0;
}
