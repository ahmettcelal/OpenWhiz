#include <iostream>
#include <fstream>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"

int main() {
    std::cout << "=== OpenWhiz High-Level API Example ===" << std::endl;

    // 1. Create a dummy CSV for testing
    std::string csv_path = "simple_data.csv";
    std::ofstream csv(csv_path);
    csv << "Feature1;Feature2;Target\n";
    csv << "0.1;0.2;0.3\n";
    csv << "0.4;0.5;0.9\n";
    csv << "0.2;0.1;0.3\n";
    csv << "0.5;0.4;0.9\n";
    csv.close();

    // 2. Initialize Neural Network
    ow::owNeuralNetwork nn;

    // 3. Load Data directly into the network
    if (nn.loadData(csv_path)) {
        std::cout << "Data loaded successfully." << std::endl;

        // 4. Create Network automatically based on data dimensions
        // Example: Hidden layer with 4 neurons (ReLU), output layer (Sigmoid)
        nn.createNeuralNetwork({4}, "ReLU", "Identity"); 
        
        std::cout << "Network architecture created automatically." << std::endl;
        std::cout << "Layers: "; nn.getLayerNames();

        // 5. High-level training
        nn.setMaximumEpochNum(100);
        nn.train();

        // 6. Predict new data
        ow::owTensor<float, 2> testInput(1, 2);
        testInput(0, 0) = 0.3f; testInput(0, 1) = 0.3f;
        auto prediction = nn.forward(testInput);

        std::cout << "\nInput: [0.3, 0.3]" << std::endl;
        std::cout << "Prediction: " << prediction(0, 0) << " (Expected near 0.6)" << std::endl;

    } else {
        std::cout << "Failed to load data." << std::endl;
    }

    return 0;
}
