#include <iostream>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"

int main() {
    std::cout << "=== OpenWhiz Attention Layer Example ===" << std::endl;

    // 1. Setup Attention Layer
    // Dimension of each feature vector is 4
    size_t featureDim = 4;
    ow::owAttentionLayer attention(featureDim);

    // 2. Create a batch of input data (3 samples)
    // Sample 0 and 2 will be similar, Sample 1 will be different
    ow::owTensor<float, 2> input(3, featureDim);
    
    // Sample 0: [1, 0, 1, 0]
    input(0, 0) = 1.0f; input(0, 1) = 0.0f; input(0, 2) = 1.0f; input(0, 3) = 0.0f;
    
    // Sample 1: [0, 1, 0, 1]
    input(1, 0) = 0.0f; input(1, 1) = 1.0f; input(1, 2) = 0.0f; input(1, 3) = 1.0f;
    
    // Sample 2: [1, 0.1, 0.9, 0] (Similar to Sample 0)
    input(2, 0) = 1.0f; input(2, 1) = 0.1f; input(2, 2) = 0.9f; input(2, 3) = 0.0f;

    std::cout << "\nInput Batch (3 samples, 4 dimensions):" << std::endl;
    input.print();

    // 3. Forward Pass through Attention
    // The layer computes attention weights between all samples in the batch
    // and produces a weighted sum of inputs.
    auto output = attention.forward(input);

    std::cout << "\nOutput after Attention Layer:" << std::endl;
    output.print();

    std::cout << "\nMechanism Explanation:" << std::endl;
    std::cout << "- The layer calculated 'how much each sample relates to others'." << std::endl;
    std::cout << "- Sample 0 and 2 have high dot-products, so they attend to each other more." << std::endl;
    std::cout << "- The output for each sample is a context-aware version of itself," << std::endl;
    std::cout << "  blended with information from other relevant samples in the batch." << std::endl;

    return 0;
}
