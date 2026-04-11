/*
 * owIdentityActivation.hpp
 *
 *  Created on: Dec 16, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owActivation.hpp"

namespace ow {

/**
 * @class owIdentityActivation
 * @brief Implements the Identity activation function (f(x) = x).
 * 
 * The Identity activation returns the input unchanged. It is typically used in the output 
 * layer of regression models where the target range is not restricted (e.g., predicting 
 * physical values in industrial sensors or computer vision coordinates).
 * 
 * Unlike Sigmoid or Tanh, it does not squash values, preserving the original scale 
 * of the data.
 */
class owIdentityActivation : public owActivation {
public:
    /**
     * @brief Forward pass: returns input as is.
     * @param input Input tensor.
     * @return Identical tensor to input.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override { return input; }

    /**
     * @brief Backward pass: gradient is 1.0, so it returns outputGradient unchanged.
     * @param input Original input.
     * @param outputGradient Incoming gradient from the next layer.
     * @return The same outputGradient.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& input, const owTensor<float, 2>& outputGradient) override { return outputGradient; }

    /**
     * @brief Deep copy of the Identity activation.
     * @return Shared pointer to new owIdentityActivation instance.
     */
    std::shared_ptr<owActivation> clone() const override { return std::make_shared<owIdentityActivation>(); }
};

} // namespace ow
