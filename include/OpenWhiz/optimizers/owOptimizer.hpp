/*
 * owLoss.hpp
 *
 *  Created on: Dec 16, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "../core/owTensor.hpp"
#include <map>
#include <vector>
#include <deque>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace ow {

class owNeuralNetwork;
class owDataset;

/**
 * @class owOptimizer
 * @brief Base class for all optimization algorithms in OpenWhiz.
 * 
 * The optimizer is responsible for updating the model parameters (weights and biases) 
 * to minimize the loss function. OpenWhiz supports two main paradigms:
 * 1. **Layer-wise (First-Order):** Updates parameters based on local gradients (e.g., SGD, Adam).
 * 2. **Global (Second-Order):** Optimizes the entire network structure simultaneously (e.g., L-BFGS).
 * 
 * @note In **embedded and mobile systems**, first-order optimizers are generally preferred 
 * due to their lower memory footprint and computational simplicity. 
 * @note For **industrial high-precision modeling**, second-order methods may provide 
 * faster convergence and better accuracy, albeit at a higher memory cost.
 */
class owOptimizer {
public:
    virtual ~owOptimizer() = default;

    /**
     * @brief Updates a set of parameters using their corresponding gradients.
     * 
     * This is the primary interface for first-order optimizers. It modifies the 'params' 
     * tensor in-place based on the 'gradients' and the internal state of the optimizer.
     *
     * @param params The tensor of weights or biases to be updated.
     * @param gradients The tensor of gradients calculated during backpropagation.
     */
    virtual void update(owTensor<float, 2>& params, const owTensor<float, 2>& gradients) = 0;

    /**
     * @brief Sets the learning rate (step size) for the optimizer.
     * @param lr The new learning rate value.
     */
    virtual void setLearningRate(float lr) { m_learningRate = lr; }

    /**
     * @brief Retrieves the current learning rate.
     * @return float The current learning rate.
     */
    virtual float getLearningRate() const { return m_learningRate; }

    /**
     * @brief Creates a deep copy of the optimizer.
     * @return std::shared_ptr<owOptimizer> A shared pointer to the cloned instance.
     */
    virtual std::shared_ptr<owOptimizer> clone() const = 0;

    /**
     * @brief Returns the name of the optimizer for serialization.
     */
    virtual std::string getOptimizerName() const = 0;

    /**
     * @brief Sets the threshold for gradient clipping.
     * 
     * Gradient clipping is a technique to prevent "exploding gradients" by capping 
     * the norm of the gradient tensor.
     * 
     * @param threshold The maximum allowed L2 norm. Set to -1 to disable clipping.
     */
    void setGradientClipThreshold(float threshold) { m_gradientClipThreshold = threshold; }
    
    /**
     * @brief Clips gradients to prevent numerical instability.
     * 
     * Uses L2 norm clipping: if ||gradients|| > threshold, then 
     * gradients = gradients * (threshold / ||gradients||).
     * 
     * @param gradients The input gradient tensor.
     * @return owTensor<float, 2> The clipped (or original) gradient tensor.
     */
    owTensor<float, 2> clipGradients(const owTensor<float, 2>& gradients) const {
        if (m_gradientClipThreshold <= 0) return gradients;
        
        float sumSq = 0;
        const float* d = gradients.data();
        for(size_t i = 0; i < gradients.size(); ++i) sumSq += d[i] * d[i];
        float norm = std::sqrt(sumSq);
        
        if (norm > m_gradientClipThreshold) {
            float scale = m_gradientClipThreshold / (norm + 1e-7f);
            return gradients * scale; 
        }
        return gradients;
    }

    /**
     * @brief Global optimization interface for Second-Order methods.
     * 
     * Unlike `update()`, this method operates on the entire network and dataset 
     * at once, allowing for complex algorithms like L-BFGS or Conjugate Gradient.
     * 
     * @param nn Pointer to the neural network to optimize.
     * @param ds Pointer to the dataset used for training.
     * @throws std::runtime_error if not implemented by the specific optimizer.
     */
    virtual void optimizeGlobal(owNeuralNetwork* nn, owDataset* ds);

    /**
     * @brief Checks if the optimizer supports global (second-order) optimization.
     * @return true if global optimization is supported, false otherwise.
     */
    virtual bool supportsGlobalOptimization() const { return false; }

protected:
    float m_learningRate = 0.01f;
    float m_gradientClipThreshold = -1.0f;

    /**
     * @brief Internal state buffers for stateful optimizers (e.g., Momentum, Adam).
     * 
     * Maps parameter pointers and buffer indices to their respective state tensors.
     */
    std::map<std::pair<void*, int>, owTensor<float, 2>> m_buffers;

    /**
     * @brief Retrieves or initializes a state buffer for a specific parameter.
     * 
     * @param paramPtr Pointer to the parameter tensor's data.
     * @param shape The shape required for the buffer tensor.
     * @param bufferIdx Index of the buffer (e.g., 0 for velocity, 1 for variance).
     * @return owTensor<float, 2>& Reference to the state buffer.
     */
    owTensor<float, 2>& getBuffer(void* paramPtr, const typename owTensor<float, 2>::owTensorShape& shape, int bufferIdx = 0) {
        auto key = std::make_pair(paramPtr, bufferIdx);
        if (m_buffers.find(key) == m_buffers.end()) {
            m_buffers[key] = owTensor<float, 2>(shape);
            m_buffers[key].setZero();
        }
        return m_buffers[key];
    }

    /**
     * @brief Copies internal buffers to another optimizer instance.
     * Used during cloning to preserve training state.
     * @param other Pointer to the target optimizer.
     */
    void copyBuffersTo(owOptimizer* other) const {
        other->m_buffers = m_buffers;
    }
};

} // namespace ow

#include "../core/owNeuralNetwork.hpp"

namespace ow {

/**
 * @brief Implementation of optimizeGlobal.
 * Must be 'inline' because this is a header-only library.
 * Now 'nn' is a complete type, so member access is allowed.
 */
inline void owOptimizer::optimizeGlobal(owNeuralNetwork* nn, owDataset* ds) {
    // If a specific optimizer (like L-BFGS) doesn't override this,
    // it will throw an error to notify the developer.
    throw std::runtime_error("Global optimization (Second-Order) is not implemented for the selected optimizer.");
}

} // namespace ow

