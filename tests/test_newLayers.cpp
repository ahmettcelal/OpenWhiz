#include "OpenWhiz/openwhiz.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

void testRescalingLayer() {
    std::cout << "Testing owRescalingLayer..." << std::endl;
    ow::owRescalingLayer layer(2.0f, 3.0f); // y = 2x + 3
    
    ow::owTensor<float, 2> input({1, 2});
    input(0, 0) = 1.0f; input(0, 1) = 2.0f;
    
    auto output = layer.forward(input);
    // 1*2+3 = 5, 2*2+3 = 7
    assert(std::abs(output(0, 0) - 5.0f) < 1e-5);
    assert(std::abs(output(0, 1) - 7.0f) < 1e-5);
    
    ow::owTensor<float, 2> grad({1, 2});
    grad(0, 0) = 1.0f; grad(0, 1) = 1.0f;
    auto inGrad = layer.backward(grad);
    // dy/dx = a = 2
    assert(std::abs(inGrad(0, 0) - 2.0f) < 1e-5);
    assert(std::abs(inGrad(0, 1) - 2.0f) < 1e-5);
    
    std::cout << "owRescalingLayer passed!" << std::endl;
}

void testAffineLayer() {
    std::cout << "Testing owAffineLayer (Training)..." << std::endl;
    ow::owAffineLayer layer; // y = 1x + 0 initially
    ow::owSGDOptimizer opt(0.1f);
    layer.setOptimizer(&opt);
    
    // Target: y = 3x + 5
    // Training data: x=1, y=8
    ow::owTensor<float, 2> input({1, 1});
    input(0, 0) = 1.0f;
    
    float initialA = layer.getA();
    float initialB = layer.getB();
    assert(initialA == 1.0f);
    assert(initialB == 0.0f);

    for(int i=0; i<100; ++i) {
        auto out = layer.forward(input);
        float lossGrad = out(0, 0) - 8.0f; // Simple MSE gradient (pred - target)
        ow::owTensor<float, 2> grad({1, 1});
        grad(0, 0) = lossGrad;
        
        layer.backward(grad);
        layer.train();
    }
    
    std::cout << "After 100 epochs: a=" << layer.getA() << ", b=" << layer.getB() << std::endl;
    assert(layer.getA() > initialA);
    assert(layer.getB() > initialB);
    
    std::cout << "owAffineLayer passed!" << std::endl;
}

int main() {
    testRescalingLayer();
    testAffineLayer();
    std::cout << "All new layer tests passed!" << std::endl;
    return 0;
}
