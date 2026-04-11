#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"

void testClusterLayer() {
    std::cout << "Testing owClusterLayer..." << std::endl;
    size_t inputSize = 2;
    size_t numClusters = 2;
    ow::owClusterLayer layer(inputSize, numClusters);
    
    // Set manual centroids for predictable output
    // Cluster 0 at (0, 0), Cluster 1 at (10, 10)
    // We need to use fromXML or access members, but since centroids are private,
    // let's use the XML path to set them or just rely on Random and check logic.
    std::string xml = 
        "<InputSize>2</InputSize>\n"
        "<NumClusters>2</NumClusters>\n"
        "<Centroids>0.0 0.0 10.0 10.0</Centroids>\n";
    layer.fromXML(xml);

    ow::owTensor<float, 2> input({1, 2});
    input(0, 0) = 0.0f; input(0, 1) = 3.0f;
    
    auto output = layer.forward(input);
    // Dist to (0,0): sqrt(0^2 + 3^2) = 3
    // Dist to (10,10): sqrt(10^2 + 7^2) = sqrt(100 + 49) = 12.2065
    assert(std::abs(output(0, 0) - 3.0f) < 1e-3);
    assert(std::abs(output(0, 1) - 12.2065f) < 1e-3);
    
    // Test backward (gradients)
    ow::owTensor<float, 2> grad({1, 2});
    grad(0, 0) = 1.0f; grad(0, 1) = 0.0f; // Only care about dist to first cluster
    auto inGrad = layer.backward(grad);
    // dDist/dx = (x - c) / dist = (0 - 0) / 3 = 0
    // dDist/dy = (y - c) / dist = (3 - 0) / 3 = 1
    assert(std::abs(inGrad(0, 0) - 0.0f) < 1e-5);
    assert(std::abs(inGrad(0, 1) - 1.0f) < 1e-5);

    std::cout << "owClusterLayer passed!" << std::endl;
}

void testPCALayer() {
    std::cout << "Testing owPCALayer..." << std::endl;
    // Create data along the line y = x (45 degrees)
    // Points: (1,1), (2,2), (3,3), (4,4), (5,5)
    ow::owTensor<float, 2> data({5, 2});
    for(size_t i=0; i<5; ++i) {
        data(i, 0) = (float)(i + 1);
        data(i, 1) = (float)(i + 1);
    }

    ow::owPrincipalComponentAnalysisLayer pca(2, 1); // 2D to 1D
    pca.fit(data);

    // The first principal component should be [1/sqrt(2), 1/sqrt(2)] or [-1/sqrt(2), -1/sqrt(2)]
    // Since it's y=x
    auto projected = pca.forward(data);
    
    std::cout << "Projected 1D values:" << std::endl;
    projected.print();
    
    // Variance should be preserved in the 1st component
    // Check if points are distinct in 1D
    for(size_t i=1; i<5; ++i) {
        assert(std::abs(projected(i, 0) - projected(i-1, 0)) > 0.1f);
    }

    // Test Reconstruction (Backward)
    ow::owTensor<float, 2> grad1D({5, 1});
    grad1D.setConstant(1.0f);
    auto grad2D = pca.backward(grad1D);
    // Back projection should be along the same vector [0.707, 0.707]
    assert(std::abs(grad2D(0, 0) - grad2D(0, 1)) < 1e-5);

    std::cout << "owPCALayer passed!" << std::endl;
}

int main() {
    try {
        testClusterLayer();
        testPCALayer();
        std::cout << "\nAll clustering and PCA tests passed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
