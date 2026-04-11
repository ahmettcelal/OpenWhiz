/*
 * owActivation.hpp
 *
 *  Created on: Dec 16, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "../core/owTensor.hpp"
#include <cmath>
#include <algorithm>
#include "../core/owSimd.hpp"

namespace ow {

/**
 * @class owActivation
 * @brief Base class for all non-linear activation functions in OpenWhiz.
 * 
 * Activation functions introduce non-linearity into the neural network, allowing it to learn
 * complex patterns. In industrial and embedded applications, choosing the right activation
 * can balance computational cost and model accuracy.
 */
class owActivation {
public:
    virtual ~owActivation() = default;

    /**
     * @brief Computes the forward pass of the activation function.
     * @param input The input tensor (typically the result of a linear layer).
     * @return A tensor of the same shape as input, with the activation applied element-wise.
     */
    virtual owTensor<float, 2> forward(const owTensor<float, 2>& input) = 0;

    /**
     * @brief Computes the backward pass (gradients) of the activation function.
     * @param input The original input tensor from the forward pass.
     * @param outputGradient The gradient of the loss with respect to the activation output.
     * @return The gradient of the loss with respect to the activation input.
     * 
     * This function implements the chain rule: dLoss/dInput = dLoss/dOutput * dOutput/dInput.
     */
    virtual owTensor<float, 2> backward(const owTensor<float, 2>& input, const owTensor<float, 2>& outputGradient) = 0;

    /**
     * @brief Creates a deep copy of the activation instance.
     * @return A shared pointer to the cloned activation.
     */
    virtual std::shared_ptr<owActivation> clone() const = 0;
};

} // namespace ow
