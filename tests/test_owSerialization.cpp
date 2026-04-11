#include "OpenWhiz/openwhiz.hpp"
#include <iostream>
#include <memory>
#include <cassert>

void testSerialization() {
    std::cout << "Testing Neural Network XML Serialization..." << std::endl;

    // 1. Create and configure a network
    ow::owNeuralNetwork nnOriginal;
    
    auto norm = std::make_shared<ow::owNormalizationLayer>(2);
    ow::owTensor<float, 2> min({1, 2}), max({1, 2});
    min(0,0)=0; min(0,1)=0; max(0,0)=100; max(0,1)=100;
    norm->setStatistics(min, max);
    nnOriginal.addLayer(norm);

    auto linear = std::make_shared<ow::owLinearLayer>(2, 1);
    linear->setActivation(std::make_shared<ow::owSigmoidActivation>());
    nnOriginal.addLayer(linear);

    nnOriginal.getOptimizer()->setLearningRate(0.05f);

    // 2. Save to XML
    std::string filename = "test_network.xml";
    nnOriginal.saveToXML(filename);
    std::cout << "Network saved to " << filename << std::endl;

    // 3. Load into a new network
    ow::owNeuralNetwork nnLoaded;
    nnLoaded.loadFromXML(filename);
    std::cout << "Network loaded from " << filename << std::endl;

    // 4. Verification
    // Check metadata
    assert(nnLoaded.getLayers().size() == nnOriginal.getLayers().size());
    assert(nnLoaded.getOptimizer()->getLearningRate() == 0.05f);
    assert(nnLoaded.getLayers()[0]->getLayerName() == "Normalization Layer");
    assert(nnLoaded.getLayers()[1]->getLayerName() == "Linear Layer");

    // Check prediction consistency
    ow::owTensor<float, 2> input({1, 2});
    input(0, 0) = 50.0f; input(0, 1) = 50.0f;

    auto predOriginal = nnOriginal.forward(input);
    auto predLoaded = nnLoaded.forward(input);

    std::cout << "Original Prediction: " << predOriginal(0, 0) << std::endl;
    std::cout << "Loaded Prediction:   " << predLoaded(0, 0) << std::endl;

    assert(std::abs(predOriginal(0, 0) - predLoaded(0, 0)) < 1e-6);

    std::cout << "Serialization test passed successfully!" << std::endl;
}

int main() {
    try {
        testSerialization();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
