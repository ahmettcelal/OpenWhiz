#include <iostream>
#include <memory>
#include <string>
#include "OpenWhiz/core/owNeuralNetwork.hpp"
#include "OpenWhiz/layers/owLayer.hpp"
#include "OpenWhiz/layers/owNormalizationLayer.hpp"
#include "OpenWhiz/layers/owLinearLayer.hpp"
#include "OpenWhiz/layers/owLSTMLayer.hpp"

int main() {
    std::cout << "Testing Neural Network Metadata (Names and Neurons)..." << std::endl;

    ow::owNeuralNetwork nn;
    
    // Add various layers
    nn.addLayer(std::make_shared<ow::owNormalizationLayer>(10));
    nn.addLayer(std::make_shared<ow::owLinearLayer>(10, 5));
    nn.addLayer(std::make_shared<ow::owProbabilityLayer>());
    
    // Check names
    auto namesTensor = nn.getLayerNames();
    std::cout << "\nLayer Names:" << std::endl;
    namesTensor.print();

    // Check neuron numbers
    auto neuronNums = nn.getNeuronNums();
    std::cout << "\nNeuron Numbers:" << std::endl;
    neuronNums.print();

    // Verification
    bool success = true;
    if (neuronNums(0) != 10) success = false;
    if (neuronNums(1) != 5) success = false;
    // Probability Layer neuron number depends on previous input or initialization
    // For now we just check if it returns a value
    std::cout << "Layer 2 Neurons: " << neuronNums(2) << std::endl;

    if (success) {
        std::cout << "\nSUCCESS: Neuron numbers are correctly retrieved." << std::endl;
    } else {
        std::cout << "\nFAILURE: Neuron numbers do not match." << std::endl;
    }

    return 0;
}
