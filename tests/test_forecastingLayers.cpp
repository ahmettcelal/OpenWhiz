#include <iostream>
#include <cassert>
#include <cmath>
#include "OpenWhiz/openwhiz.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void testPositionEncoding() {
    std::cout << "Testing owPositionEncodingLayer..." << std::endl;
    size_t maxLen = 10;
    size_t dModel = 4;
    ow::owPositionEncodingLayer layer(maxLen, dModel);

    ow::owTensor<float, 2> input({5, 4});
    input.setZero();
    
    auto output = layer.forward(input);
    
    // Check if output is not zero for non-zero positions/features
    for(size_t i=1; i<5; ++i) { // Skip pos 0 as sin(0) is 0
        bool allZero = true;
        for(size_t j=0; j<4; ++j) if(output(i, j) != 0.0f) allZero = false;
        assert(!allZero);
    }
    
    // Check periodicity/consistency: pos 0 should have sin(0)=0, cos(0)=1
    assert(std::abs(output(0, 0) - 0.0f) < 1e-5);
    assert(std::abs(output(0, 1) - 1.0f) < 1e-5);

    std::cout << "owPositionEncodingLayer passed!" << std::endl;
}

void testDateTimeEncoding() {
    std::cout << "Testing owDateTimeEncodingLayer..." << std::endl;
    ow::owDateTimeEncodingLayer layer;

    // Input: [Hour, DayOfWeek, Month, DayOfMonth]
    ow::owTensor<float, 2> input({2, 4});
    input(0, 0) = 0.0f;  input(0, 1) = 0.0f; input(0, 2) = 1.0f; input(0, 3) = 1.0f;  // Midnight, Monday, Jan, 1st
    input(1, 0) = 12.0f; input(1, 1) = 3.0f; input(1, 2) = 7.0f; input(1, 3) = 15.0f; // Noon, Thursday, July, 15th

    auto output = layer.forward(input);

    assert(output.shape()[1] == 8);

    // Test Midnight (Hour 0)
    // sin(0)=0, cos(0)=1
    assert(std::abs(output(0, 0) - 0.0f) < 1e-5);
    assert(std::abs(output(0, 1) - 1.0f) < 1e-5);

    // Test Noon (Hour 12)
    // sin(PI)=0, cos(PI)=-1
    assert(std::abs(output(1, 0) - 0.0f) < 1e-5);
    assert(std::abs(output(1, 1) - (-1.0f)) < 1e-5);

    std::cout << "owDateTimeEncodingLayer passed!" << std::endl;
}

int main() {
    testPositionEncoding();
    testDateTimeEncoding();
    std::cout << "All forecasting layer tests passed!" << std::endl;
    return 0;
}
