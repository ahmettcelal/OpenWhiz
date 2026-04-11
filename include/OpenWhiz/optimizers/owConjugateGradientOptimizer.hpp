/*
 * owConjugateGradientOptimizer.hpp
 *
 *  Created on: Jan 12, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owOptimizer.hpp"

namespace ow {

/**
 * @class owConjugateGradientOptimizer
 * @brief Conjugate Gradient (CG) optimizer.
 * 
 * CG is an algorithm for the numerical solution of particular systems of linear 
 * equations or non-linear optimization. It is an intermediate between 
 * First-Order (SGD) and Second-Order (Newton) methods.
 * 
 * Instead of moving only in the direction of the negative gradient, it moves 
 * in a direction "conjugate" to the previous search directions.
 * 
 * **Advantages:**
 * - **Superior Stability:** Less prone to oscillations than standard SGD.
 * - **Faster Convergence:** Often reaches the minimum faster than Momentum on 
 *   well-behaved quadratic surfaces.
 * - **Memory Efficient:** Requires only two additional buffers per parameter, 
 *   making it a viable "higher-order" alternative for systems with moderate RAM.
 * 
 * @note In **industrial control systems**, CG is favored for its predictable 
 * convergence properties when modeling physical processes.
 */
class owConjugateGradientOptimizer : public owOptimizer {
public:
    /**
     * @brief Constructs a Conjugate Gradient optimizer.
     * @param lr The initial learning rate (default: 0.01).
     */
    owConjugateGradientOptimizer(float lr = 0.01f) { m_learningRate = lr; }

    /**
     * @brief Performs a Conjugate Gradient update step.
     * 
     * Calculates the new search direction based on the current gradient and the 
     * previous direction using the Polak-Ribière formula.
     *
     * @param params The parameter tensor to update.
     * @param gradients The gradient tensor.
     */
    void update(owTensor<float, 2>& params, const owTensor<float, 2>& gradients) override {
        auto g_clipped = clipGradients(gradients);
        auto& d = getBuffer(&params, params.shape(), 0); // Search direction
        auto& prevG = getBuffer(&params, params.shape(), 1); // Previous gradient

        float dotG = 0.0f, dotPrevG = 0.0f, dotGPrevDiff = 0.0f;
        for (size_t i = 0; i < params.size(); ++i) {
            float g = g_clipped.data()[i];
            float pg = prevG.data()[i];
            dotG += g * g;
            dotPrevG += pg * pg;
            dotGPrevDiff += g * (g - pg);
        }

        float beta = (dotPrevG < 1e-10f) ? 0.0f : std::max(0.0f, dotGPrevDiff / dotPrevG);

        for (size_t i = 0; i < params.size(); ++i) {
            d.data()[i] = -g_clipped.data()[i] + beta * d.data()[i];
            params.data()[i] += m_learningRate * d.data()[i];
            prevG.data()[i] = g_clipped.data()[i];
        }
    }

    /**
     * @brief Returns the name of the optimizer.
     */
    std::string getOptimizerName() const override { return "Conjugate Gradient"; }

    /**
     * @brief Creates a deep copy of the Conjugate Gradient optimizer.
     * @return std::shared_ptr<owOptimizer> A shared pointer to the cloned instance.
     */
    std::shared_ptr<owOptimizer> clone() const override {
        auto copy = std::make_shared<owConjugateGradientOptimizer>(m_learningRate);
        copy->m_gradientClipThreshold = m_gradientClipThreshold;
        copyBuffersTo(copy.get());
        return copy;
    }
};

} // namespace ow
