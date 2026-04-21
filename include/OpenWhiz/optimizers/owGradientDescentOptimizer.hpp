/*
 * owGradientDescentOptimizer.hpp
 *
 *  Created on: Apr 21, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owOptimizer.hpp"

namespace ow {

/**
 * @class owGradientDescentOptimizer
 * @brief Vanilla Gradient Descent (GD) optimizer.
 * 
 * Gradient Descent is the classic optimization algorithm for minimizing loss.
 * In its "Vanilla" form, it moves parameters in the opposite direction of the 
 * calculated gradient.
 * 
 * Formula: θ = θ - η * ∇J(θ)
 * 
 * **Characteristics:**
 * - **Simplicity:** No momentum, no adaptive learning rates.
 * - **Stability:** When used with the full dataset (Batch GD), it provides 
 *   a stable and monotonic convergence path.
 * - **Foundation:** Serves as the mathematical baseline for all other optimizers.
 */
class owGradientDescentOptimizer : public owOptimizer {
public:
    /**
     * @brief Constructs a Gradient Descent optimizer.
     * @param lr The initial learning rate (default: 0.01).
     */
    owGradientDescentOptimizer(float lr = 0.01f) { m_learningRate = lr; }

    /**
     * @brief Performs a standard Gradient Descent update step.
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
    std::string getOptimizerName() const override { return "GD"; }

    /**
     * @brief Creates a deep copy of the Gradient Descent optimizer.
     * @return std::shared_ptr<owOptimizer> A shared pointer to the cloned instance.
     */
    std::shared_ptr<owOptimizer> clone() const override {
        auto copy = std::make_shared<owGradientDescentOptimizer>(m_learningRate);
        copy->m_gradientClipThreshold = m_gradientClipThreshold;
        copyBuffersTo(copy.get());
        return copy;
    }
};

} // namespace ow
