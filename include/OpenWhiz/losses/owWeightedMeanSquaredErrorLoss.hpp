/*
 * owWeightedMeanSquaredErrorLoss.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLoss.hpp"

namespace ow {

/**
 * @class owWeightedMeanSquaredErrorLoss
 * @brief Computes the Weighted Mean Squared Error (WMSE) loss.
 * 
 * WMSE is a variation of MSE where each sample's contribution to the total 
 * loss is scaled by a corresponding weight.
 * 
 * @details
 * Mathematical formula:
 * \f$ L = \frac{1}{n} \sum_{i=1}^{n} w_i (y_i - \hat{y}_i)^2 \f$
 * where \f$ w_i \f$ is the weight for the \f$ i \f$-th sample.
 * 
 * Features:
 * - Importance Sampling: Allows the model to focus on more "important" samples.
 * - Handling Imbalance: Can be used to counteract datasets where some ranges 
 *   of values are underrepresented.
 * 
 * Comparison with other losses:
 * - Use WMSE when you have external knowledge that some data points are 
 *   more reliable or critical than others.
 * - Use standard MSE if all samples are equally important.
 * 
 * Platform notes:
 * - Computer: Standard for handling large, noisy datasets with reliability scores.
 * - Mobile/Web: Useful for on-device personalization where user-confirmed 
 *   labels have higher weight.
 * - Industrial: Critical when certain operating conditions (e.g., extreme heat) 
 *   must be modeled more accurately than others.
 */
class owWeightedMeanSquaredErrorLoss : public owLoss {
public:
    /**
     * @brief Sets the weights for each sample in the batch.
     * @param weights A tensor containing weights of the same shape as predictions.
     */
    void setWeights(const owTensor<float, 2>& weights) { m_weights = weights; }

    /**
     * @brief Computes the WMSE loss value.
     * @param prediction Predicted values.
     * @param target Ground truth values.
     * @return Weighted average squared difference.
     * @note If weight tensor size does not match, returns 0.0f.
     */
    float compute(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        float loss = 0.0f;
        size_t n = prediction.size();
        if (m_weights.size() != n) return 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float diff = prediction.data()[i] - target.data()[i];
            loss += m_weights.data()[i] * diff * diff;
        }
        return loss / static_cast<float>(n);
    }

    /**
     * @brief Computes the gradient of the WMSE loss.
     * 
     * Derivative with respect to prediction \f$ \hat{y}_i \f$:
     * \f$ \frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n} w_i (\hat{y}_i - y_i) \f$
     * 
     * @param prediction Predicted values.
     * @param target Ground truth values.
     * @return Gradient tensor.
     */
    owTensor<float, 2> gradient(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        owTensor<float, 2> grad(prediction.shape());
        size_t n = prediction.size();
        float factor = 2.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            grad.data()[i] = factor * m_weights.data()[i] * (prediction.data()[i] - target.data()[i]);
        }
        return grad;
    }

    /**
     * @brief Returns the name of the loss function.
     */
    std::string getLossName() const override { return "Weighted Mean Squared Error Loss"; }

    /**
     * @brief Creates a deep copy of the WMSE loss function.
     * @return A shared pointer to the cloned object.
     */
    std::shared_ptr<owLoss> clone() const override {
        auto copy = std::make_shared<owWeightedMeanSquaredErrorLoss>();
        copy->m_weights = m_weights;
        return copy;
    }
private:
    owTensor<float, 2> m_weights;
};

} // namespace ow
