/*
 * owPinballLoss.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLoss.hpp"

namespace ow {

/**
 * @class owPinballLoss
 * @brief Computes the Pinball loss, primarily used for Quantile Regression.
 * 
 * Pinball loss (also known as Quantile loss) is used when you want to predict 
 * a specific quantile of the target distribution rather than the mean or median.
 * 
 * @details
 * Mathematical formula:
 * \f$ L_\tau(y, \hat{y}) = \begin{cases} \tau(y - \hat{y}) & \text{if } y \ge \hat{y} \\ (1 - \tau)(\hat{y} - y) & \text{if } y < \hat{y} \end{cases} \f$
 * where \f$ \tau \f$ (tau) is the target quantile (e.g., 0.5 for the median).
 * 
 * Features:
 * - Allows for interval forecasting by predicting different quantiles (e.g., 
 *   0.05 and 0.95 for a 90% confidence interval).
 * - Asymmetric: Penalizes overestimation and underestimation differently 
 *   depending on the quantile.
 * 
 * Comparison with other losses:
 * - Use Pinball Loss for probabilistic forecasting and uncertainty estimation.
 * - MAE is a special case of Pinball Loss where \f$ \tau = 0.5 \f$ (median regression).
 * 
 * Platform notes:
 * - Computer: Vital for supply chain and energy demand forecasting.
 * - Mobile/Web: Useful for displaying "ranges" rather than single point estimates.
 * - Industrial: Critical for safety margins where underestimating a value 
 *   (like pressure) is more dangerous than overestimating it.
 */
class owPinballLoss : public owLoss {
public:
    /**
     * @brief Constructs Pinball loss for a specific quantile.
     * @param quantile The target quantile \f$ \tau \f$ in the range (0, 1).
     */
    owPinballLoss(float quantile = 0.5f) : m_quantile(quantile) {}

    /**
     * @brief Computes the Pinball loss value.
     * @param prediction Predicted values.
     * @param target Ground truth values.
     * @return Average pinball loss.
     */
    float compute(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        float loss = 0.0f;
        size_t n = prediction.size();
        for (size_t i = 0; i < n; ++i) {
            float diff = target.data()[i] - prediction.data()[i];
            if (diff >= 0) loss += m_quantile * diff;
            else loss -= (1.0f - m_quantile) * diff;
        }
        return loss / static_cast<float>(n);
    }

    /**
     * @brief Computes the gradient of the Pinball loss.
     * 
     * Derivative with respect to prediction \f$ \hat{y}_i \f$:
     * \f$ \frac{\partial L}{\partial \hat{y}_i} = \begin{cases} -\tau / n & \text{if } y \ge \hat{y} \\ (1 - \tau) / n & \text{if } y < \hat{y} \end{cases} \f$
     * 
     * @param prediction Predicted values.
     * @param target Ground truth values.
     * @return Gradient tensor.
     */
    owTensor<float, 2> gradient(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        owTensor<float, 2> grad(prediction.shape());
        size_t n = prediction.size();
        float factor = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            float diff = target.data()[i] - prediction.data()[i];
            if (diff >= 0) grad.data()[i] = -factor * m_quantile;
            else grad.data()[i] = factor * (1.0f - m_quantile);
        }
        return grad;
    }

    /**
     * @brief Returns the name of the loss function.
     */
    std::string getLossName() const override { return "Pinball Loss"; }

    /**
     * @brief Creates a deep copy of the Pinball loss function.
     * @return A shared pointer to the cloned object.
     */
    std::shared_ptr<owLoss> clone() const override { return std::make_shared<owPinballLoss>(m_quantile); }
private:
    float m_quantile;
};

} // namespace ow
