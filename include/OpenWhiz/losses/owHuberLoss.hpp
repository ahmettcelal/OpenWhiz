/*
 * owHuberLoss.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLoss.hpp"

namespace ow {

/**
 * @class owHuberLoss
 * @brief Computes the Huber loss, a robust loss for regression.
 * 
 * Huber loss combines the best properties of MSE and MAE. It is quadratic for 
 * small errors and linear for large errors, making it less sensitive to 
 * outliers than MSE while remaining smooth near zero.
 * 
 * @details
 * Mathematical formula:
 * \f$ L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\ \delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases} \f$
 * 
 * Features:
 * - Robustness: More robust to outliers than MSE.
 * - Stability: Differentiable at zero, unlike MAE, providing more stable convergence.
 * - Tuneable: The 'delta' parameter defines the threshold between quadratic 
 *   and linear regions.
 * 
 * Comparison with other losses:
 * - Use Huber loss when you want a smooth loss that is robust to outliers.
 * - Use MSE if you want to penalize all errors quadratically.
 * - Use MAE if you want a constant penalty for all error magnitudes.
 * 
 * Platform notes:
 * - Computer: Excellent default for regression where outlier behavior is unknown.
 * - Mobile/Web: Slightly more complex to compute than MSE/MAE due to branching.
 * - Industrial: Good for sensors where occasional noise spikes should be dampened.
 */
class owHuberLoss : public owLoss {
public:
    /**
     * @brief Constructs Huber loss with a specific delta threshold.
     * @param delta The threshold at which the loss transitions from quadratic to linear.
     */
    owHuberLoss(float delta = 1.0f) : m_delta(delta) {}

    /**
     * @brief Computes the Huber loss value.
     * @param prediction Predicted output tensor.
     * @param target Ground truth tensor.
     * @return Computed Huber loss.
     */
    float compute(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        float loss = 0.0f;
        size_t n = prediction.size();
        for (size_t i = 0; i < n; ++i) {
            float diff = std::abs(prediction.data()[i] - target.data()[i]);
            if (diff <= m_delta) loss += 0.5f * diff * diff;
            else loss += m_delta * (diff - 0.5f * m_delta);
        }
        return loss / static_cast<float>(n);
    }

    /**
     * @brief Computes the gradient of the Huber loss.
     * 
     * Derivative with respect to prediction \f$ \hat{y}_i \f$:
     * \f$ \frac{\partial L}{\partial \hat{y}_i} = \begin{cases} \frac{1}{n}(\hat{y}_i - y_i) & \text{for } |\hat{y}_i - y_i| \le \delta \\ \frac{1}{n} \cdot \delta \cdot \text{sgn}(\hat{y}_i - y_i) & \text{otherwise} \end{cases} \f$
     * 
     * @param prediction Predicted output tensor.
     * @param target Ground truth tensor.
     * @return Gradient tensor.
     */
    owTensor<float, 2> gradient(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        owTensor<float, 2> grad(prediction.shape());
        size_t n = prediction.size();
        float factor = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            float diff = prediction.data()[i] - target.data()[i];
            if (std::abs(diff) <= m_delta) grad.data()[i] = factor * diff;
            else grad.data()[i] = factor * m_delta * (diff > 0 ? 1.0f : -1.0f);
        }
        return grad;
    }

    /**
     * @brief Returns the name of the loss function.
     */
    std::string getLossName() const override { return "Huber Loss"; }

    /**
     * @brief Creates a deep copy of the Huber loss function.
     * @return A shared pointer to the cloned object.
     */
    std::shared_ptr<owLoss> clone() const override { return std::make_shared<owHuberLoss>(m_delta); }

private:
    float m_delta;
};

} // namespace ow
