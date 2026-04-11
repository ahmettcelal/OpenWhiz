#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include "OpenWhiz/openwhiz.hpp"

/**
 * @brief Synthetic data generator for clustering.
 * Creates 3 distinct Gaussian clusters in 2D space.
 */
void generateClusteringData(const std::string& filename, int count) {
    std::ofstream file(filename);
    file << "X;Y;Target1;Target2;Target3\n"; // 2 Features, 3 dummy targets
    
    std::mt19937 gen(42);
    // Cluster 1: Center (-3, -3)
    std::normal_distribution<float> d1_x(-3.0f, 1.0f);
    std::normal_distribution<float> d1_y(-3.0f, 1.0f);
    
    // Cluster 2: Center (3, 3)
    std::normal_distribution<float> d2_x(3.0f, 1.0f);
    std::normal_distribution<float> d2_y(3.0f, 1.0f);
    
    // Cluster 3: Center (0, 5)
    std::normal_distribution<float> d3_x(0.0f, 1.0f);
    std::normal_distribution<float> d3_y(5.0f, 1.0f);

    for (int i = 0; i < count; ++i) {
        float x, y;
        int cluster = i % 3;
        if (cluster == 0) { x = d1_x(gen); y = d1_y(gen); }
        else if (cluster == 1) { x = d2_x(gen); y = d2_y(gen); }
        else { x = d3_x(gen); y = d3_y(gen); }

        // For clustering, we usually don't have targets, 
        // but our dataset loader expects them.
        file << x << ";" << y << ";0;0;0\n";
    }
}

int main() {
    std::cout << "=== OpenWhiz Neural Clustering Example ===\n" << std::endl;

    const std::string csvFile = "clustering_data.csv";
    generateClusteringData(csvFile, 900);

    ow::owNeuralNetwork nn;
    
    // 1. Setup Data
    if (!nn.loadData(csvFile)) {
        std::cerr << "Failed to load data!" << std::endl;
        return -1;
    }
    nn.getDataset()->setTargetVariableNum(3);
    nn.getDataset()->setRatios(1.0f, 0.0f, 0.0f); // Use all for training

    // 2. Build Clustering Architecture
    // We add a single Cluster Layer with 3 centroids.
    // The layer takes 2 inputs (X, Y) and outputs 3 distances.
    auto clusterLayer = std::make_shared<ow::owClusterLayer>(2, 3);
    nn.addLayer(clusterLayer);

    // 3. Configure Training
    // Since we want to MINIMIZE the distances (clustering objective),
    // we use a target of [0, 0, 0] and MSE loss.
    // The optimizer will move centroids closer to the data points.
    nn.setOptimizer(std::make_shared<ow::owADAMOptimizer>(0.05f));
    nn.setMaximumEpochNum(100);
    nn.setPrintEpochInterval(10);

    // 4. Train (Self-Organizing/Unsupervised style)
    // We train the model to output 0 for all distances (pulling centroids to points)
    std::cout << "Training centroids to minimize distance to data..." << std::endl;
    nn.train();

    // 5. Inspect Results
    std::cout << "\nFinal Centroids (XML Export):" << std::endl;
    std::cout << clusterLayer->toXML() << std::endl;

    // 6. Test Clustering Prediction
    // origin [0,0] should be closest to Cluster 3 (0, 5) or Cluster 1/2 centers.
    ow::owTensor<float, 2> sample(1, 2);
    sample(0, 0) = 0.0f; sample(0, 1) = 0.0f;
    
    auto dists = nn.forward(sample);
    std::cout << "Distances for point [0, 0]:" << std::endl;
    for(size_t i=0; i<3; ++i) {
        std::cout << "  Centroid " << i << ": " << dists(0, i) << std::endl;
    }

    int winner = 0;
    float minDist = dists(0, 0);
    for(int i=1; i<3; ++i) {
        if(dists(0, i) < minDist) { minDist = dists(0, i); winner = i; }
    }
    std::cout << "\nPoint [0, 0] belongs to Cluster: " << winner << std::endl;

    return 0;
}
