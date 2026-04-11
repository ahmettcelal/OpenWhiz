
#include "../../include/OpenWhiz/openwhiz.hpp"
#include <iostream>
#include <vector>

int main() {
    try {
        std::cout << "--- OpenWhiz Clustering Example ---" << std::endl;

        // 1. Create a dummy dataset for clustering
        // 4 features, 3 clusters (so 3 target variables representing distances to centroids)
        std::ofstream csv("examples/clusteringExample/clustering_data.csv");
        csv << "f1,f2,f3,f4,d1,d2,d3\n";
        for (int i = 0; i < 100; ++i) {
            // Cluster 1 around (1,1,1,1)
            csv << 1.0+((rand()%100)/500.0) << "," << 1.0+((rand()%100)/500.0) << "," 
                << 1.0+((rand()%100)/500.0) << "," << 1.0+((rand()%100)/500.0) << ",0,0,0\n";
            // Cluster 2 around (-1,-1,-1,-1)
            csv << -1.0+((rand()%100)/500.0) << "," << -1.0+((rand()%100)/500.0) << "," 
                << -1.0+((rand()%100)/500.0) << "," << -1.0+((rand()%100)/500.0) << ",0,0,0\n";
            // Cluster 3 around (5,5,0,0)
            csv << 5.0+((rand()%100)/500.0) << "," << 5.0+((rand()%100)/500.0) << "," 
                << 0.0+((rand()%100)/500.0) << "," << 0.0+((rand()%100)/500.0) << ",0,0,0\n";
        }
        csv.close();

        auto dataset = std::make_shared<ow::owDataset>();
        dataset->setDelimiter(',');
        if (!dataset->loadFromCSV("examples/clusteringExample/clustering_data.csv")) {
            std::cerr << "Failed to load data!" << std::endl;
            return 1;
        }
        dataset->setTargetVariableNum(3); // 3 targets, remaining are inputs (4)

        // 2. Setup Neural Network
        ow::owNeuralNetwork nn;
        nn.setDataset(dataset);
        
        // Structure: Normalization -> Projection(4->2) -> Cluster(2->3) -> Distance(3->3)
        std::vector<int> hidden = { 2 }; 
        nn.createNeuralNetwork(ow::owProjectType::CLUSTERING, hidden);

        std::cout << "Model Architecture:" << std::endl;
        auto names = nn.getLayerNames();
        for (size_t i = 0; i < names.size(); ++i) {
            std::cout << "Layer " << i << ": " << names(i) << std::endl;
        }

        // 3. Training
        nn.setMaximumEpochNum(200);
        nn.setPrintEpochInterval(50);
        nn.getOptimizer()->setLearningRate(0.01f);
        
        std::cout << "\nStarting Training..." << std::endl;
        nn.train();
        std::cout << "Training finished. Reason: " << nn.getTrainingFinishReason() << std::endl;

        // 4. Inference
        ow::owTensor<float, 2> sample(1, 4);
        sample.setValues({1.1f, 0.9f, 1.05f, 1.0f}); // Near Cluster 1
        auto result = nn.forward(sample);
        
        std::cout << "\nDistances for sample near Cluster 1:" << std::endl;
        result.print();

        sample.setValues({4.9f, 5.1f, 0.05f, -0.05f}); // Near Cluster 3
        result = nn.forward(sample);
        std::cout << "\nDistances for sample near Cluster 3:" << std::endl;
        result.print();

        // 5. Serialization Test
        std::cout << "\nSaving model to clustering_model.xml..." << std::endl;
        nn.saveToXML("clustering_model.xml");

        ow::owNeuralNetwork nn2;
        std::cout << "Loading model from clustering_model.xml..." << std::endl;
        if (nn2.loadFromXML("examples/clusteringExample/clustering_model.xml")) {
             std::cout << "Load successful. Layers in loaded model:" << std::endl;
             auto names2 = nn2.getLayerNames();
             for (size_t i = 0; i < names2.size(); ++i) {
                 std::cout << "Layer " << i << ": " << names2(i) << std::endl;
             }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
