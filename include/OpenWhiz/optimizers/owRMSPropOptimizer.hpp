/*
 * owRMSPropOptimizer.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owOptimizer.hpp"

namespace ow {

/**
 * @class owRMSPropOptimizer
 * @brief Root Mean Square Propagation (RMSProp) optimizer.
 * 
 * RMSProp is an adaptive learning rate method designed to resolve Adagrad's 
 * radically diminishing learning rates. It maintains a moving average of the 
 * squared gradients and divides the learning rate by the root of this average.
 * 
 * Formula:
 * E[g^2]_t = γ * E[g^2]_{t-1} + (1 - γ) * g_t^2
 * θ = θ - (η / sqrt(E[g^2]_t + ε)) * g_t
 * 
 * **Advantages:**
 * - **Handles Non-Stationary Objectives:** Excellent for recurrent neural networks (RNNs).
 * - **Adaptive Step Sizes:** Automatically adjusts learning rates for each parameter.
 * - **Moderate Memory:** Requires one additional memory buffer per parameter.
 * 
 * @note In **industrial time-series forecasting**, RMSProp often provides better 
 * stability than Adam for certain types of seasonal data.
 */
class owRMSPropOptimizer : public owOptimizer {
public:
    /**
     * @brief Constructs an RMSProp optimizer.
     * @param lr The initial learning rate (default: 0.01).
     * @param decay The discount factor for the history (default: 0.9).
     * @param eps A small constant for numerical stability (default: 1e-8).
     */
    owRMSPropOptimizer(float lr = 0.01f, float decay = 0.9f, float eps = 1e-8f)
        : m_decay(decay), m_epsilon(eps) { m_learningRate = lr; }

    /**
     * @brief Performs an RMSProp update step.
     * 
     * Updates the squared gradient moving average and then updates the parameters.
     *
     * @param params The parameter tensor to update.
     * @param gradients The gradient tensor.
     */
    void update(owTensor<float, 2>& params, const owTensor<float, 2>& gradients) override {
        auto g_clipped = clipGradients(gradients);
        auto& s = getBuffer(&params, params.shape());
        for (size_t i = 0; i < params.size(); ++i) {
            float g = g_clipped.data()[i];
            s.data()[i] = m_decay * s.data()[i] + (1.0f - m_decay) * g * g;
            params.data()[i] -= m_learningRate * g / (std::sqrt(s.data()[i]) + m_epsilon);
        }
    }

    /**
     * @brief Returns the name of the optimizer.
     */
    std::string getOptimizerName() const override { return "RMSProp"; }

    /**
     * @brief Creates a deep copy of the RMSProp optimizer.
     * @return std::shared_ptr<owOptimizer> A shared pointer to the cloned instance.
     */
    std::shared_ptr<owOptimizer> clone() const override {
        auto copy = std::make_shared<owRMSPropOptimizer>(m_learningRate, m_decay, m_epsilon);
        copy->m_gradientClipThreshold = m_gradientClipThreshold;
        copyBuffersTo(copy.get());
        return copy;
    }
private:
    float m_decay, m_epsilon;
};

} // namespace ow
