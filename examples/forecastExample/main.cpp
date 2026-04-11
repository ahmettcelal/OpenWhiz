#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Generates a sine wave CSV for forecasting (2 columns for explicit input/target)
void generateSineWaveCSV(const std::string& filename, int points) {
    std::ofstream file(filename);
    file << "Input,Target\n";
    for (int i = 0; i < points; ++i) {
        float val = std::sin(2.0f * M_PI * i / 50.0f); // Period of 50 points
        float nextVal = std::sin(2.0f * M_PI * (i + 1) / 50.0f);
        file << val << "," << nextVal << "\n";
    }
    file.close();
}

int main() {
    std::cout << "=== OpenWhiz Time Series Forecasting Example ===\n" << std::endl;

    // 1. Generate sine wave data
    const std::string csvFile = "examples/forecastExample/sine_wave.csv";
    generateSineWaveCSV(csvFile, 300);
    std::cout << "Generated 300 points of sine wave data." << std::endl;

    // 2. Setup Dataset
    auto dataset = std::make_shared<ow::owDataset>();
    dataset->setDelimiter(',');
    if (!dataset->loadFromCSV(csvFile)) {
        std::cerr << "Failed to load CSV data." << std::endl;
        return -1;
    }
    
    // 1 column as target (the last one), which means 1 column left for input
    dataset->setTargetVariableNum(1);
    dataset->setRatios(0.8f, 0.1f, 0.1f);

    std::cout << "Input variables: " << dataset->getInputVariableNum() << std::endl;
    std::cout << "Target variables: " << dataset->getTargetVariableNum() << std::endl;

    // 3. Initialize Neural Network with FORECASTING project type
    ow::owNeuralNetwork nn;
    nn.setDataset(dataset);
    
    // Automatically configures: 
    // Normalization -> SlidingWindowLayer (10-step history + current) -> Linear(16) -> Linear(1) -> InverseNormalization
    const int windowSize = 10;
    nn.createNeuralNetwork(ow::owProjectType::FORECASTING, {16}, windowSize);
    
    nn.setOptimizer(std::make_shared<ow::owLBFGSOptimizer>(1.0f));
    nn.setMaximumEpochNum(200);
    nn.setLossStagnationTolerance(1e-6f);
    nn.setPrintEpochInterval(100);

    // 4. Train
    std::cout << "Training model to predict sine wave..." << std::endl;
    nn.train();

    // 5. Evaluation
    auto testIn = dataset->getTestInput();
    auto testOut = dataset->getTestTarget();
    auto eval = nn.evaluatePerformance(testIn, testOut, 0.05f);
    
    std::cout << "\nFinal Performance on Test Set:" << std::endl;
    nn.printEvaluationReport(eval);

    // 6. Forecast Test (Real-time mode)
    std::cout << "\n--- Rolling Prediction Test (Last 5 points) ---" << std::endl;
    
    auto fullIn = dataset->getTrainInput();
    size_t startIdx = 100;
    
    for (size_t i = 0; i < 5; ++i) {
        ow::owTensor<float, 2> currentSample(1, 1);
        currentSample(0, 0) = fullIn(startIdx + i, 0);

        auto pred = nn.forward(currentSample);
        float actualNext = fullIn(startIdx + i + 1, 0); // Roughly, next value

        std::cout << "Step " << i << ": Predicted: " << pred(0, 0) 
                  << " | Actual Next: " << actualNext 
                  << " | Err: " << std::abs(pred(0, 0) - actualNext) << std::endl;
    }

    return 0;
}
