#include <iostream>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"

int main() {
    std::cout << "=== OpenWhiz Multi-Head Attention Example ===" << std::endl;

    // 1. Setup Multi-Head Attention Layer
    // dModel = 8 (Input dimension)
    // numHeads = 2 (Number of parallel attention specialists)
    // Each head will work on a subspace of dimension 4 (8 / 2)
    size_t dModel = 8;
    size_t numHeads = 2;
    ow::owMultiHeadAttentionLayer mha(dModel, numHeads);

    // 2. Create a batch of input data (4 samples, 8 dimensions each)
    ow::owTensor<float, 2> input(4, dModel);
    input.setRandom(-1.0f, 1.0f);

    std::cout << "\nInput Batch (4 samples, 8 dimensions):" << std::endl;
    input.print();

    // 3. Forward Pass through Multi-Head Attention
    // This will:
    // a) Project input into Q, K, V using learned weights.
    // b) Split into 2 heads.
    // c) Calculate attention scores in parallel for each head.
    // d) Combine and project back to dModel dimension.
    auto output = mha.forward(input);

    std::cout << "\nOutput after Multi-Head Attention (Same shape as input):" << std::endl;
    output.print();

    std::cout << "\nMechanism Explanation:" << std::endl;
    std::cout << "- dModel (8) was split into " << numHeads << " heads of size 4." << std::endl;
    std::cout << "- Head 1 focused on one set of relationships within the 8D space." << std::endl;
    std::cout << "- Head 2 focused on a different set of relationships simultaneously." << std::endl;
    std::cout << "- The result is a richer, 'multi-perspective' representation of the data." << std::endl;

    return 0;
}
