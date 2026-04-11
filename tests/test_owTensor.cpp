#include <iostream>
#include <cassert>
#include "OpenWhiz/owTensor.hpp"

int main() {
    std::cout << "Testing owTensor new methods..." << std::endl;

    ow::owTensor<float, 2>::owTensorShape shape = {2, 2};
    ow::owTensor<float, 2> tensor(shape);

    // Test setZero
    tensor.setConstant(5.0f);
    tensor.setZero();
    for (size_t i = 0; i < tensor.size(); ++i) {
        assert(tensor.data()[i] == 0.0f);
    }
    std::cout << "setZero passed." << std::endl;

    // Test setConstant
    tensor.setConstant(3.14f);
    for (size_t i = 0; i < tensor.size(); ++i) {
        assert(tensor.data()[i] == 3.14f);
    }
    std::cout << "setConstant passed." << std::endl;

    // Test setRandom
    tensor.setRandom(10.0f, 20.0f);
    for (size_t i = 0; i < tensor.size(); ++i) {
        assert(tensor.data()[i] >= 10.0f && tensor.data()[i] <= 20.0f);
    }
    std::cout << "setRandom passed." << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
