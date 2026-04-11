#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"


int main() {
    std::cout << "=== OpenWhiz Airfoil Self Noise Approximation Example ===" << std::endl;

    const std::string csvFile = "examples/airfoilExample/airfoil_self_noise.csv";

    ow::owNeuralNetwork nn;

    // 1. Setup Data
    nn.getDataset()->setRatios(0.6f, 0.2f, 0.2f);

    if (!nn.loadData(csvFile)) {
        std::cerr << "Failed to load data!" << std::endl;
        return -1;
    }

    // 2. Build Network Architecture
    // More capacity for complex non-linear airfoil physics
    nn.createNeuralNetwork({32, 16}, "ReLU", "Identity", true);
    
    // Mean Squared Error provides much faster convergence near the minimum for L-BFGS
    nn.setLoss(std::make_shared<ow::owMeanSquaredErrorLoss>());
    nn.setRegularization(ow::NONE);

    // 3. Configure Training
    nn.setOptimizer(std::make_shared<ow::owLBFGSOptimizer>(1.0f)); 
    nn.setMaximumEpochNum(1000); 
    nn.setMinimumError(0.001f); 
    nn.setPrintEpochInterval(1); 
    nn.setLossStagnationEnabled(false);
    nn.setLossStagnationTolerance(0.0005f);
    nn.setLossStagnationPatience(10);

    // 4. Train
    nn.train();

    // 5. Final Evaluation
    auto testIn = nn.getDataset()->getTestInput();
    auto testOut = nn.getDataset()->getTestTarget();
    auto eval = nn.evaluatePerformance(testIn, testOut, 0.05f); // 5% tolerance
    
    std::cout << "\nFinal Performance on Test Set:" << std::endl;
    nn.printEvaluationReport(eval);

    // Logging extra stats before manual prediction
    std::cout << "Total Epochs: " << nn.getTrainingEpochNum() << std::endl;
    std::cout << "Total Time: " << nn.getTrainingTime() << "s" << std::endl;
    std::cout << "Final Train Error: " << nn.getLastTrainError() << std::endl;
    std::cout << "Final Val Error: " << nn.getLastValError() << std::endl;
    std::cout << "First Sample Type: " << nn.getDataset()->getSampleTypeString(0) << std::endl;

    // 6. Manual Prediction Check (Using RAW values directly!)
    // Input features: frequency;angle_of_attack;chord_lenght;velocity;thickness
    ow::owTensor<float, 2> input(1, 5);
    input.setValues({800.0f, 0.0f, 0.3048f, 71.3f, 0.00266337f});
    
    auto pred = nn.forward(input);
    
    std::cout << "Prediction for [800, 0, 0.3048, 71.3, 0.00266337] = " << pred(0, 0) << " (Actual: 126.201)" << std::endl;

    return 0;
}
