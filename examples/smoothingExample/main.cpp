#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include "OpenWhiz/openwhiz.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    std::cout << "=== OpenWhiz Exponential Smoothing Example ===\n" << std::endl;

    // 1. Generate Noisy Sine Wave Data
    const int numPoints = 20;
    ow::owTensor<float, 2> noisyData(numPoints, 1);
    
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> noise(-0.2f, 0.2f);

    std::cout << "Generating data (Sine + Random Noise)..." << std::endl;
    for (int i = 0; i < numPoints; ++i) {
        float cleanVal = std::sin(2.0f * M_PI * i / 10.0f);
        noisyData(i, 0) = cleanVal + noise(gen);
    }

    // 2. Setup Smoothing Layer
    // Alpha = 0.3 means: 30% current value, 70% previous trend
    float alpha = 0.3f;
    auto smoothingLayer = std::make_shared<ow::owSmoothingLayer>(alpha);

    std::cout << "Applying Exponential Smoothing (Alpha: " << alpha << ")..." << std::endl;
    
    // 3. Process data through the layer
    ow::owTensor<float, 2> smoothedData = smoothingLayer->forward(noisyData);

    // 4. Compare Results
    std::cout << "\n" << std::setw(10) << "Step" << std::setw(15) << "Noisy" << std::setw(15) << "Smoothed" << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    for (int i = 0; i < numPoints; ++i) {
        std::cout << std::setw(10) << i 
                  << std::setw(15) << std::fixed << std::setprecision(4) << noisyData(i, 0)
                  << std::setw(15) << smoothedData(i, 0) << std::endl;
    }

    std::cout << "\nObservation:" << std::endl;
    std::cout << "- Notice how 'Noisy' values jump around." << std::endl;
    std::cout << "- 'Smoothed' values follow a much more stable curve." << std::endl;
    std::cout << "- This helps Neural Networks focus on the trend rather than the noise." << std::endl;

    return 0;
}
