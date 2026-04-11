/*
 * owMomentumOptimizer.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owOptimizer.hpp"
#include <algorithm>

namespace ow {

/**
 * @class owMomentumOptimizer
 * @brief SGD with Momentum optimizer.
 * 
 * Momentum addresses the problem of SGD oscillating in "ravines" where the surface 
 * curves more steeply in one dimension than another. It introduces a "velocity" 
 * term that accumulates the past gradients.
 * 
 * Formula:
 * v_t = γ * v_{t-1} + η * ∇J(θ)
 * θ = θ - v_t
 * 
 * **Advantages:**
 * - **Faster Convergence:** Accelerates training in areas of low curvature.
 * - **Dampens Oscillations:** Stabilizes training in narrow ravines.
 * - **Low Memory Overhead:** Requires only one additional memory buffer per 
 *   parameter (storing the velocity).
 * 
 * @note This is a balanced choice for **mobile platforms** that need faster 
 * training than SGD but still have moderate memory constraints.
 */
class owMomentumOptimizer : public owOptimizer {
public:
    /**
     * @brief Constructs a Momentum optimizer.
     * @param lr The initial learning rate (default: 0.01).
     * @param momentum The momentum factor (default: 0.9). Range: [0, 1].
     */
    owMomentumOptimizer(float lr = 0.01f, float momentum = 0.9f) : m_momentum(momentum) { m_learningRate = lr; }

    /**
     * @brief Performs a Momentum update step.
     * 
     * Calculates the new velocity based on the previous velocity and current gradient, 
     * then updates the parameters.
     *
     * @param params The parameter tensor to update.
     * @param gradients The gradient tensor.
     */
    void update(owTensor<float, 2>& params, const owTensor<float, 2>& gradients) override {
        auto g_clipped = clipGradients(gradients);
        auto& v = getBuffer(&params, params.shape());
        for (size_t i = 0; i < params.size(); ++i) {
            v.data()[i] = m_momentum * v.data()[i] + m_learningRate * g_clipped.data()[i];
            params.data()[i] -= v.data()[i];
        }
    }

    /**
     * @brief Returns the name of the optimizer.
     */
    std::string getOptimizerName() const override { return "Momentum"; }

    /**
     * @brief Creates a deep copy of the Momentum optimizer.
     * @return std::shared_ptr<owOptimizer> A shared pointer to the cloned instance.
     */
    std::shared_ptr<owOptimizer> clone() const override {
        auto copy = std::make_shared<owMomentumOptimizer>(m_learningRate, m_momentum);
        copy->m_gradientClipThreshold = m_gradientClipThreshold;
        copyBuffersTo(copy.get());
        return copy;
    }
private:
    float m_momentum;
};

} // namespace ow
