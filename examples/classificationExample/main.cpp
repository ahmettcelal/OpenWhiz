#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"

// Synthetic classification problem: 3 regions in 2D space
// Region 0: Center circle (r < 2)
// Region 1: Outer ring (2 <= r < 4)
// Region 2: Far corners (r >= 4)
void generateClassificationData(const std::string& filename, int count) {
    std::ofstream file(filename);
    file << "X;Y;Class0;Class1;Class2\n"; 
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    for (int i = 0; i < count; ++i) {
        float x = dist(gen);
        float y = dist(gen);
        float r = std::sqrt(x*x + y*y);
        
        int cls = (r < 2.0f) ? 0 : (r < 4.0f ? 1 : 2);

        file << x << ";" << y << ";" 
             << (cls == 0 ? 1 : 0) << ";" 
             << (cls == 1 ? 1 : 0) << ";" 
             << (cls == 2 ? 1 : 0) << "\n";
    }
}

int main() {
    std::cout << "=== OpenWhiz Multi-Class Classification Example ===\n" << std::endl;

    const std::string csvFile = "examples/classificationExample/classification_data.csv";
    generateClassificationData(csvFile, 2000); // More data for better boundary learning

    ow::owNeuralNetwork nn;
    
    // 1. Setup Data
    // Enable auto-normalization at dataset level for reliable [0,1] mapping
    nn.getDataset()->setAutoNormalizeEnabled(true);
    if (!nn.loadData(csvFile)) {
        std::cerr << "Failed to load data!" << std::endl;
        return -1;
    }
    nn.getDataset()->setTargetVariableNum(3); 
    nn.getDataset()->setRatios(0.8f, 0.1f, 0.1f);

    // 2. Build Network Architecture
    // 2 Inputs -> 64 Hidden -> 32 Hidden -> 3 Outputs
    // No internal normalization layer because we normalized at dataset level
    nn.createNeuralNetwork({64, 32}, "ReLU", "Identity", false);
    nn.addLayer(std::make_shared<ow::owProbabilityLayer>());
    
    // 3. Configure Training
    nn.setLoss(std::make_shared<ow::owCategoricalCrossEntropyLoss>());
    nn.setOptimizer(std::make_shared<ow::owLBFGSOptimizer>(1.0f));
    nn.setMaximumEpochNum(500);
    nn.setMinimumError(0.001f);
    nn.setPrintEpochInterval(50);
    nn.setLossStagnationTolerance(1e-6f);
    nn.setLossStagnationPatience(30);

    // 4. Train
    nn.train();

    // 5. Custom Classification Evaluation (Argmax Accuracy)
    auto testIn = nn.getDataset()->getTestInput();
    auto testOut = nn.getDataset()->getTestTarget();
    auto pred = nn.forward(testIn);
    
    int correct = 0;
    for (size_t i = 0; i < testIn.shape()[0]; ++i) {
        int predCls = 0, trueCls = 0;
        float maxP = -1.0f, maxT = -1.0f;
        for (int j = 0; j < 3; ++j) {
            if (pred(i, j) > maxP) { maxP = pred(i, j); predCls = j; }
            if (testOut(i, j) > maxT) { maxT = testOut(i, j); trueCls = j; }
        }
        if (predCls == trueCls) correct++;
    }
    
    std::cout << "\nFinal Performance on Test Set:" << std::endl;
    std::cout << "Argmax Accuracy: " << (float)correct / testIn.shape()[0] * 100.0f << "%" << std::endl;

    // 6. Test Specific Points
    auto testPoint = [&](float x, float y, const std::string& label) {
        ow::owTensor<float, 2> sample(1, 2);
        // Map raw input to [0,1] using dataset params
        auto px = nn.getDataset()->getNormalizationParams(0);
        auto py = nn.getDataset()->getNormalizationParams(1);
        sample(0, 0) = (x - px.first) / (px.second - px.first);
        sample(0, 1) = (y - py.first) / (py.second - py.first);

        auto p = nn.forward(sample);
        std::cout << "Prediction for " << label << " [" << x << "," << y << "]: "
                  << "C0:" << p(0,0) << " C1:" << p(0,1) << " C2:" << p(0,2) << std::endl;
    };

    testPoint(0.0f, 0.0f, "Origin (Class 0)");
    testPoint(2.5f, 0.0f, "Ring (Class 1)");
    testPoint(4.5f, 4.5f, "Corner (Class 2)");

    return 0;
}
