#include <iostream>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"

int main() {
    std::cout << "=== OpenWhiz Anomaly Detection Project Example ===" << std::endl;

    // 1. Load Dataset
    auto dataset = std::make_shared<ow::owDataset>();
    dataset->setDelimiter(',');
    if (!dataset->loadFromCSV("anomaly_data.csv")) {
        std::cerr << "Failed to load anomaly_data.csv" << std::endl;
        return 1;
    }
    
    // We want to use all columns as inputs for anomaly detection
    dataset->setTargetVariableNum(0); 

    std::cout << "Dataset loaded with " << dataset->getInputVariableNum() << " features." << std::endl;

    // 2. Setup Neural Network with ANOMALY_DETECTION project type
    ow::owNeuralNetwork nn;
    nn.setDataset(dataset);

    // Structure: Normalization -> Projection(3->2) -> Anomaly Detection
    // We provide {2} as hiddenSizes to set the latent dimension of the projection layer to 2.
    std::vector<int> hidden = { 2 };
    nn.createNeuralNetwork(ow::owProjectType::ANOMALY_DETECTION, hidden);

    std::cout << "\nModel Architecture:" << std::endl;
    auto names = nn.getLayerNames();
    for (size_t i = 0; i < names.size(); ++i) {
        std::cout << "Layer " << i << ": " << names(i) << std::endl;
    }

    // 3. Test the network with some data
    std::cout << "\nTesting with normal data..." << std::endl;
    ow::owTensor<float, 2> normalTest(1, 3);
    normalTest(0, 0) = 10.0f;
    normalTest(0, 1) = 20.0f;
    normalTest(0, 2) = 30.0f;
    
    auto outputNormal = nn.forward(normalTest);
    std::cout << "Output for normal data (Projected):" << std::endl;
    outputNormal.print();

    std::cout << "\nTesting with anomalous data (Feature 1 is 100.0)..." << std::endl;
    ow::owTensor<float, 2> anomalyTest(1, 3);
    anomalyTest(0, 0) = 100.0f; 
    anomalyTest(0, 1) = 20.0f;
    anomalyTest(0, 2) = 30.0f;

    auto outputAnomaly = nn.forward(anomalyTest);
    std::cout << "Output for anomalous data (Anomaly Detection should suppress/flag):" << std::endl;
    outputAnomaly.print();

    // 4. Serialization Test
    std::cout << "\nSaving model to anomaly_model.xml..." << std::endl;
    if (nn.saveToXML("anomaly_model.xml")) {
        std::cout << "Model saved successfully." << std::endl;
    }

    // 5. Load and verify
    ow::owNeuralNetwork nnLoaded;
    if (nnLoaded.loadFromXML("anomaly_model.xml")) {
        std::cout << "Model loaded successfully from XML." << std::endl;
        std::cout << "Loaded Project Type ID: " << static_cast<int>(nnLoaded.getProjectType()) << std::endl;
        
        auto loadedNames = nnLoaded.getLayerNames();
        std::cout << "Loaded Model Architecture:" << std::endl;
        for (size_t i = 0; i < loadedNames.size(); ++i) {
            std::cout << "Layer " << i << ": " << loadedNames(i) << std::endl;
        }
    }

    return 0;
}
