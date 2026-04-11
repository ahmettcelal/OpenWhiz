#include "OpenWhiz/openwhiz.hpp"
#include <iostream>
#include <memory>

int main() {
    std::cout << "Testing OpenWhiz Neural Network with Activations..." << std::endl;

    // 1. Create a network
    ow::owNeuralNetwork nn;
    
    // Add a hidden layer: 2 inputs -> 4 neurons, with ReLU activation
    auto hiddenLayer = std::make_shared<ow::owLinearLayer>(2, 4);
    hiddenLayer->setActivation(std::make_shared<ow::owReLUActivation>());
    nn.addLayer(hiddenLayer);

    // Add an output layer: 4 neurons -> 1 output, with Sigmoid activation
    auto outputLayer = std::make_shared<ow::owLinearLayer>(4, 1);
    outputLayer->setActivation(std::make_shared<ow::owSigmoidActivation>());
    nn.addLayer(outputLayer);

    // 2. Dummy Input and Target
    ow::owTensor<float, 2> input({1, 2});
    input.data()[0] = 0.5f;
    input.data()[1] = -0.2f;

    ow::owTensor<float, 2> target({1, 1});
    target.data()[0] = 0.8f; // Target output

    // 3. Training Loop
    std::cout << "Starting training loop (500 steps)..." << std::endl;
    nn.getOptimizer()->setLearningRate(0.1f);
    
    for (int i = 0; i <= 500; ++i) {
        auto prediction = nn.forward(input);
        float loss = nn.getLoss()->compute(prediction, target);
        nn.backward(prediction, target);
        nn.trainStep();

        if (i % 100 == 0) {
            std::cout << "Step " << i << ", Loss: " << loss << ", Prediction: " << prediction.data()[0] << std::endl;
        }
    }

    auto finalPrediction = nn.forward(input);
    std::cout << "\nFinal Prediction: " << finalPrediction.data()[0] << " (Target: 0.8)" << std::endl;

    return 0;
}
