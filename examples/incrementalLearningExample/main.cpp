#include <iostream>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"

int main() {
    std::cout << "=== OpenWhiz Incremental Learning Example ===" << std::endl;

    // 1. Initialize Neural Network
    ow::owNeuralNetwork nn;
    
    // Create a dummy dataset to define dimensions
    auto dataset = std::make_shared<ow::owDataset>();
    nn.setDataset(dataset);

    // Initial architecture: 2 inputs -> 4 hidden -> 1 output
    // We manually set dataset metadata for architecture creation
    dataset->prepareForecastingData(2, 1, 0); // (Dummy call just to set sizes internally if needed)
    
    // Better: let's just setup a simple structure manually for this example
    auto l1 = std::make_shared<ow::owLinearLayer>(2, 4);
    l1->setActivation(std::make_shared<ow::owReLUActivation>());
    auto l2 = std::make_shared<ow::owLinearLayer>(4, 1);
    l2->setActivation(std::make_shared<ow::owIdentityActivation>());
    
    nn.addLayer(l1);
    nn.addLayer(l2);

    std::cout << "Initial Status (isPartiallyFitted): " << (nn.isPartiallyFitted() ? "True" : "False") << std::endl;

    // 2. First piece of data arrives (Incremental Learning)
    ow::owTensor<float, 2> input1(1, 2);
    input1(0, 0) = 0.5f; input1(0, 1) = 0.8f;
    ow::owTensor<float, 2> target1(1, 1);
    target1(0, 0) = 1.3f;

    std::cout << "\nNew data arrived. Performing partialFit..." << std::endl;
    nn.partialFit(input1, target1, 10); // Train for 10 steps on this specific sample

    std::cout << "Status after partialFit: " << (nn.isPartiallyFitted() ? "True" : "False") << std::endl;

    // 3. Save the model
    std::string modelPath = "incremental_model.xml";
    std::cout << "Saving model to XML..." << std::endl;
    nn.saveToXML(modelPath);

    std::cout << "Status after saveToXML: " << (nn.isPartiallyFitted() ? "True" : "False") << std::endl;

    // 4. Second piece of data arrives
    ow::owTensor<float, 2> input2(1, 2);
    input2(0, 0) = 0.1f; input2(0, 1) = 0.2f;
    ow::owTensor<float, 2> target2(1, 1);
    target2(0, 0) = 0.3f;

    std::cout << "\nMore data arrived. Performing another partialFit..." << std::endl;
    nn.partialFit(input2, target2, 5);

    std::cout << "Final Status (isPartiallyFitted): " << (nn.isPartiallyFitted() ? "True" : "False") << std::endl;

    std::cout << "\nExplanation:" << std::endl;
    std::cout << "- Model starts with 'False'." << std::endl;
    std::cout << "- partialFit sets it to 'True', indicating unsaved training progress." << std::endl;
    std::cout << "- saveToXML resets it to 'False', meaning all knowledge is now permanent." << std::endl;

    return 0;
}
