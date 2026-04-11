#include <iostream>
#include "OpenWhiz/openwhiz.hpp"

int main() {
    std::cout << "=== OpenWhiz Clipping Layer Example ===" << std::endl;

    // 1. Setup Clipping Layer with range [-1.0, 1.0]
    float minVal = -1.0f;
    float maxVal = 1.0f;
    ow::owClippingLayer clipper(minVal, maxVal);

    // 2. Create input data with some out-of-range values
    ow::owTensor<float, 2> input(1, 5);
    input(0, 0) = -5.0f; // Way below min
    input(0, 1) = -0.5f; // Within range
    input(0, 2) = 0.0f;  // Within range
    input(0, 3) = 0.8f;  // Within range
    input(0, 4) = 10.0f; // Way above max

    std::cout << "\nInput Data (contains outliers -5.0 and 10.0):" << std::endl;
    input.print();

    // 3. Forward Pass
    auto output = clipper.forward(input);

    std::cout << "\nOutput Data (Clipped to [-1.0, 1.0]):" << std::endl;
    output.print();

    // 4. Backward Pass (Gradient Masking)
    // Clipping layer should zero out gradients for clipped values
    ow::owTensor<float, 2> incomingGradient(ow::owTensor<float, 2>::owTensorShape{1, 5}, 1.0f); // Gradient of 1.0 for all
    auto inputGradient = clipper.backward(incomingGradient);

    std::cout << "\nInput Gradient (Zeroed where values were clipped):" << std::endl;
    inputGradient.print();

    std::cout << "\nExplanation:" << std::endl;
    std::cout << "- Values outside [-1.0, 1.0] were set to the boundaries." << std::endl;
    std::cout << "- Gradients for these indices are 0.0, meaning the model won't try" << std::endl;
    std::cout << "  to 'push' these values further during training when they are out of bounds." << std::endl;

    return 0;
}
