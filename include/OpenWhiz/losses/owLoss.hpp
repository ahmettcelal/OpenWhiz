/*
 * owLoss.hpp
 *
 *  Created on: Dec 16, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "../core/owTensor.hpp"
#include <cmath>
#include <algorithm>
#include <memory>

namespace ow {

/**
 * @class owLoss
 * @brief Abstract base class for all loss functions in the OpenWhiz library.
 * 
 * Loss functions are used to measure the difference between the predicted values 
 * and the actual target values. This class provides the interface for computing 
 * the loss value and its gradient, which is essential for backpropagation during 
 * neural network training.
 * 
 * @details
 * Loss functions in OpenWhiz are designed to work with 2D tensors (typically 
 * [batch_size, output_size]). The implementation follows the standard 
 * optimization paradigm where the goal is to minimize this loss value.
 * 
 * @note
 * For industrial and mobile applications, ensure that the chosen loss function 
 * aligns with the numerical stability requirements of the platform.
 */
class owLoss {
public:
    virtual ~owLoss() = default;

    /**
     * @brief Computes the scalar loss value.
     * 
     * @param prediction The tensor containing the predicted values from the network.
     * @param target The tensor containing the ground truth values.
     * @return The computed loss as a float.
     */
    virtual float compute(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) = 0;

    /**
     * @brief Computes the gradient of the loss with respect to the predictions.
     * 
     * This gradient is used to start the backpropagation process.
     * 
     * @param prediction The tensor containing the predicted values.
     * @param target The tensor containing the ground truth values.
     * @return A tensor of the same shape as 'prediction' containing the gradients.
     */
    virtual owTensor<float, 2> gradient(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) = 0;

    /**
     * @brief Returns the name of the loss function for serialization.
     * @return std::string Name (e.g., "Mean Squared Error Loss").
     */
    virtual std::string getLossName() const = 0;

    /**
     * @brief Creates a deep copy of the loss function object.
     * 
     * @return A shared pointer to the cloned owLoss object.
     */
    virtual std::shared_ptr<owLoss> clone() const = 0;
};

} // namespace ow
