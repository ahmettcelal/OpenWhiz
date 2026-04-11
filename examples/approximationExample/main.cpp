#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"

// Complex non-linear function to approximate: y = sin(x) * cos(x/2) + exp(-|x|/5)
float complexFunction(float x) {
    return std::sin(x) * std::cos(x / 2.0f) + std::exp(-std::abs(x) / 5.0f);
}

void generateData(const std::string& filename, int count) {
    std::ofstream file(filename);
    file << "Input;Target\n";
    for (int i = 0; i < count; ++i) {
        float x = -10.0f + static_cast<float>(i) * (20.0f / count);
        float y = complexFunction(x);
        file << x << ";" << y << "\n";
    }
}

int main() {
    std::cout << "=== OpenWhiz Function Approximation Example ===\n" << std::endl;

    const std::string csvFile = "examples/approximationExample/approximation_data.csv";
    generateData(csvFile, 2000);

    ow::owNeuralNetwork nn;
    
    // 1. Setup Data
    nn.getDataset()->setAutoNormalizeEnabled(true);
    nn.getDataset()->setRatios(0.7f, 0.15f, 0.15f);

    if (!nn.loadData(csvFile)) {
        std::cerr << "Failed to load data!" << std::endl;
        return -1;
    }
    
    // 2. Statistical Analysis
    ow::owDatasetStatistics stats;
    stats.setDataset(nn.getDataset());
    auto report = stats.analyzeRegressionSuitability(0, 1);
    std::cout << "Statistical Recommendation: " << report.recommendation << std::endl;
    std::cout << "Initial Correlation: " << stats.calculateCorrelation(0, 1) << std::endl;

    // 3. Build Network Architecture
    // Use Tanh for smooth gradients in regression
    nn.createNeuralNetwork({32, 32}, "Tanh", "Identity", false);
    
    // 4. Configure Training
    nn.setLoss(std::make_shared<ow::owMeanSquaredErrorLoss>());
    nn.setOptimizer(std::make_shared<ow::owADAMOptimizer>(0.01f));
    nn.setMaximumEpochNum(2000);
    nn.setPrintEpochInterval(100);
    nn.setLossStagnationTolerance(1e-7f);
    nn.setLossStagnationPatience(50);

    // 5. Train
    nn.train();

    // 6. Final Evaluation
    auto testIn = nn.getDataset()->getTestInput();
    auto testOut = nn.getDataset()->getTestTarget();
    auto eval = nn.evaluatePerformance(testIn, testOut, 0.05f); // 5% tolerance
    
    std::cout << "\nFinal Performance on Test Set:" << std::endl;
    nn.printEvaluationReport(eval);

    // 7. Manual Prediction Check
    float testX = 2.5f; // A point model hasn't specifically trained on
    // Manually normalize input based on dataset stats
    auto params = nn.getDataset()->getNormalizationParams(0);
    float normX = (testX - params.first) / (params.second - params.first);
    
    ow::owTensor<float, 2> input(1, 1);
    input(0, 0) = normX;
    
    auto pred = nn.forward(input);
    
    // Denormalize output
    auto outParams = nn.getDataset()->getNormalizationParams(1);
    float realPred = pred(0, 0) * (outParams.second - outParams.first) + outParams.first;

    std::cout << "Prediction for x = " << testX << ": " << realPred << " (Actual: " << complexFunction(testX) << ")" << std::endl;

    return 0;
}
