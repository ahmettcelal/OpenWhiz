#include "OpenWhiz/openwhiz.hpp"
#include <iostream>
#include <memory>
#include <string>

int main() {
    std::cout << "Testing Layer Names and Neural Network Metadata..." << std::endl;

    ow::owNeuralNetwork nn;
    
    // Add various layers
    nn.addLayer(std::make_shared<ow::owNormalizationLayer>(10));
    nn.addLayer(std::make_shared<ow::owLinearLayer>(10, 5));
    nn.addLayer(std::make_shared<ow::owProbabilityLayer>());

    // 1. Check individual layer names
    auto layers = nn.getLayers();
    std::cout << "Layer 0 Name: " << layers[0]->getLayerName() << std::endl;
    std::cout << "Layer 1 Name: " << layers[1]->getLayerName() << std::endl;
    std::cout << "Layer 2 Name: " << layers[2]->getLayerName() << std::endl;

    // 2. Check getLayerNames() returning owTensor<std::string, 1>
    auto namesTensor = nn.getLayerNames();
    std::cout << "\nNames from Tensor:" << std::endl;
    namesTensor.print();

    // Verification
    if (namesTensor(0) == "Normalization Layer" && 
        namesTensor(1) == "Linear Layer" && 
        namesTensor(2) == "Probability Layer") {
        std::cout << "\nSUCCESS: Layer names are correctly recorded and retrieved." << std::endl;
    } else {
        std::cout << "\nFAILURE: Layer names do not match." << std::endl;
    }

    return 0;
}
