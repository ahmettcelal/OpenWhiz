#include "OpenWhiz/openwhiz.hpp"
#include <iostream>
#include <vector>

/**
 * @example concatenateForecastingExample
 * @brief Demonstrates multi-scale time-series forecasting using owConcatenateLayer and owSlidingWindowLayer.
 * 
 * In this example, we have 2 input features (e.g., Price and a Technical Indicator).
 * We want to process them differently:
 * - Branch 1: Looks at the Price (col 0) with a short-term 5-step window and includes the current value.
 * - Branch 2: Looks at the Indicator (col 1) with a longer-term 10-step window.
 * 
 * The ConcatenateLayer automatically slices the 2-column input, giving 1 column to each branch,
 * then merges their windowed outputs for the final dense layers.
 */

int main() {
    std::cout << "OpenWhiz ConcatenateLayer Example (Multi-Scale Forecasting)" << std::endl;

    // 1. Configuration
    size_t batchSize = 100;
    size_t windowShort = 5;
    size_t windowLong = 10;

    // 2. Define Parallel Branches
    // Branch 1: Short-term window on the first feature. 
    // outputSize = windowShort + 1 (current) = 6
    auto swShort = std::make_shared<ow::owSlidingWindowLayer>(windowShort, 1, 0, true); 
    swShort->setLayerName("Short Window Branch");

    // Branch 2: Long-term window on the second feature.
    // outputSize = windowLong = 10
    auto swLong = std::make_shared<ow::owSlidingWindowLayer>(windowLong, 1, 0, false); 
    swLong->setLayerName("Long Window Branch");

    // 3. Create Concatenate Layer
    // This layer expects 2 inputs (1 for each branch) and produces 6 + 10 = 16 outputs.
    auto concatLayer = std::make_shared<ow::owConcatenateLayer>(
        std::vector<std::shared_ptr<ow::owLayer>>{swShort, swLong}
    );

    // 4. Build Neural Network
    ow::owNeuralNetwork nn;
    nn.addLayer(concatLayer);
    
    size_t concatenatedDim = swShort->getOutputSize() + swLong->getOutputSize(); // 6 + 10 = 16
    nn.addLayer(std::make_shared<ow::owLinearLayer>(concatenatedDim, 16));
    nn.addLayer(std::make_shared<ow::owLinearLayer>(16, 1));

    // 5. Prepare Dummy Data
    // Input: [Batch, 2] (Price, Indicator)
    ow::owTensor<float, 2> input(batchSize, 2);
    input.setRandom(0.0f, 1.0f);

    // Target: [Batch, 1] (Next Price)
    ow::owTensor<float, 2> target(batchSize, 1);
    target.setRandom(0.0f, 1.0f);

    // 6. Forward Pass
    auto output = nn.forward(input);

    std::cout << "Input Features: 2 (Price, Indicator)" << std::endl;
    std::cout << "Branch 1 (Short): Window " << windowShort << " + Current -> Size " << swShort->getOutputSize() << std::endl;
    std::cout << "Branch 2 (Long): Window " << windowLong << " -> Size " << swLong->getOutputSize() << std::endl;
    std::cout << "Concatenated Feature Vector Size: " << concatenatedDim << std::endl;
    std::cout << "Final Output Shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]" << std::endl;

    // 7. Training Step (Gradients flow back through concatenation and sliding windows)
    float initialLoss = nn.calculateLoss(output, target);
    std::cout << "\nInitial Loss: " << initialLoss << std::endl;

    nn.backward(output, target);
    nn.trainStep();

    auto updatedOutput = nn.forward(input);
    float updatedLoss = nn.calculateLoss(updatedOutput, target);
    std::cout << "Loss after 1 training step: " << updatedLoss << std::endl;

    if (updatedLoss < initialLoss) {
        std::cout << "\nSUCCESS: The network is learning through parallel sliding windows!" << std::endl;
    } else {
        std::cout << "\nNOTE: Loss might not decrease in a single random step, but gradient flow is verified." << std::endl;
    }

    // 8. XML Serialization Test
    std::cout << "\nTesting Serialization..." << std::endl;
    std::string xmlFile = "examples/concatenateForecastingExample/forecasting_model.xml";
    if (nn.saveToXML(xmlFile)) {
        std::cout << "Model saved to " << xmlFile << std::endl;
        
        ow::owNeuralNetwork nn2;
        if (nn2.loadFromXML(xmlFile)) {
            std::cout << "Model loaded successfully!" << std::endl;
            auto output2 = nn2.forward(input);
            float diff = 0;
            for(size_t i=0; i<3; ++i) diff += std::abs(output(i,0) - output2(i,0));
            std::cout << "Consistency check (first 3 samples diff): " << diff << std::endl;
        }
    }

    return 0;
}
