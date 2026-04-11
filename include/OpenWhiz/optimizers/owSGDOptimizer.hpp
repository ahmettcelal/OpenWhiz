/*
 * owSGDOptimizer.hpp
 *
 *  Created on: Jan 12, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owOptimizer.hpp"

namespace ow {

/**
 * @class owSGDOptimizer
 * @brief Stochastic Gradient Descent (SGD) optimizer.
 * 
 * SGD is the most fundamental optimization algorithm. It updates parameters by 
 * moving them in the opposite direction of the gradient, scaled by a learning rate.
 * 
 * Formula: θ = θ - η * ∇J(θ)
 * 
 * **Advantages:**
 * - **Minimal Memory:** Requires zero additional memory buffers beyond the parameters themselves.
 * - **High Speed:** Lowest computational overhead per update step.
 * - **Ideal for Embedded/Mobile:** Best choice for systems with extremely limited RAM or 
 *   simple microcontrollers where every byte and cycle counts.
 */
class owSGDOptimizer : public owOptimizer {
public:
    /**
     * @brief Constructs an SGD optimizer.
     * @param lr The initial learning rate (default: 0.01).
     */
    owSGDOptimizer(float lr = 0.01f) { m_learningRate = lr; }

    /**
     * @brief Performs a standard SGD update step.
     * 
     * Applies gradient clipping if enabled, then subtracts the scaled gradient 
     * from the parameters.
     *
     * @param params The parameter tensor to update.
     * @param gradients The gradient tensor.
     */
    void update(owTensor<float, 2>& params, const owTensor<float, 2>& gradients) override {
        auto g_clipped = clipGradients(gradients);
        for (size_t i = 0; i < params.size(); ++i) {
            params.data()[i] -= m_learningRate * g_clipped.data()[i];
        }
    }

    /**
     * @brief Returns the name of the optimizer.
     */
    std::string getOptimizerName() const override { return "SGD"; }

    /**
     * @brief Creates a deep copy of the SGD optimizer.
     * @return std::shared_ptr<owOptimizer> A shared pointer to the cloned instance.
     */
    std::shared_ptr<owOptimizer> clone() const override {
        auto copy = std::make_shared<owSGDOptimizer>(m_learningRate);
        copy->m_gradientClipThreshold = m_gradientClipThreshold;
        copyBuffersTo(copy.get());
        return copy;
    }
};

} // namespace ow
