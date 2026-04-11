#include <iostream>
#include <memory>
#include <cassert>
#include "OpenWhiz/core/owNeuralNetwork.hpp"
#include "OpenWhiz/layers/owNormalizationLayer.hpp"
#include "OpenWhiz/layers/owInverseNormalizationLayer.hpp"
#include "OpenWhiz/layers/owProbabilityLayer.hpp"

void testSpecialLayers() {
    std::cout << "Testing Special Layers..." << std::endl;

    // 1. Test Normalization
    ow::owNormalizationLayer norm(2);
    ow::owTensor<float, 2> min({1, 2}), max({1, 2});
    min(0, 0) = 0; min(0, 1) = 10;
    max(0, 0) = 100; max(0, 1) = 20;
    norm.setStatistics(min, max);

    ow::owTensor<float, 2> input({1, 2});
    input(0, 0) = 50; input(0, 1) = 15;
    
    auto normalized = norm.forward(input);
    std::cout << "Normalized: " << normalized(0, 0) << ", " << normalized(0, 1) << " (Expected: 0.5, 0.5)" << std::endl;
    assert(std::abs(normalized(0, 0) - 0.5f) < 1e-5);

    // 2. Test Inverse Normalization
    ow::owInverseNormalizationLayer invNorm(2);
    invNorm.setStatistics(min, max);
    auto inversed = invNorm.forward(normalized);
    std::cout << "Inversed: " << inversed(0, 0) << ", " << inversed(0, 1) << " (Expected: 50, 15)" << std::endl;
    assert(std::abs(inversed(0, 0) - 50.0f) < 1e-5);

    // 3. Test Probability (Softmax)
    ow::owProbabilityLayer prob;
    ow::owTensor<float, 2> logits({1, 3});
    logits(0, 0) = 1.0f; logits(0, 1) = 1.0f; logits(0, 2) = 1.0f;
    auto probs = prob.forward(logits);
    std::cout << "Logits: " << logits(0, 0) << ", " << logits(0, 1) << ", " << logits(0, 2) << std::endl;
    std::cout << "Probabilities: " << probs(0, 0) << ", " << probs(0, 1) << ", " << probs(0, 2) << " (Expected: 0.33, 0.33, 0.33)" << std::endl;
    assert(std::abs(probs(0, 0) - 0.3333f) < 1e-3);

    std::cout << "Special layers test passed!" << std::endl;
}

int main() {
    testSpecialLayers();
    return 0;
}
