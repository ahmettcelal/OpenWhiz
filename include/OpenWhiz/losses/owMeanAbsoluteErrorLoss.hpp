/*
 * owMeanAbsoluteErrorLoss.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLoss.hpp"

namespace ow {

/**
 * @class owMeanAbsoluteErrorLoss
 * @brief Computes the Mean Absolute Error (MAE) loss.
 * 
 * MAE loss calculates the average of the absolute differences between 
 * predictions and targets.
 * 
 * @details
 * Mathematical formula:
 * \f$ L = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \f$
 * where \f$ y_i \f$ is the target and \f$ \hat{y}_i \f$ is the prediction.
 * 
 * Features:
 * - Robust to outliers: Errors are treated linearly, so large outliers don't 
 *   disproportionately influence the loss compared to MSE.
 * - Simple interpretation as the average error magnitude.
 * 
 * Comparison with other losses:
 * - Use MAE when your dataset has significant outliers you want to ignore.
 * - Use MSE when you want to penalize outliers more heavily.
 * - Note that MAE has a non-smooth derivative at zero, which can lead to 
 *   oscillations in weight updates if learning rates are high.
 * 
 * Platform notes:
 * - Computer: Common in forecasting and scenarios where median-like estimations are preferred.
 * - Mobile/Web: Very lightweight to compute.
 * - Industrial: Useful in quality control where average deviation is the standard metric.
 */
class owMeanAbsoluteErrorLoss : public owLoss {
public:
    /**
     * @brief Computes the MAE loss value.
     * @param prediction Predicted output tensor.
     * @param target Ground truth tensor.
     * @return The average absolute difference.
     */
    float compute(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        float loss = 0.0f;
        size_t n = prediction.size();
        for (size_t i = 0; i < n; ++i) loss += std::abs(prediction.data()[i] - target.data()[i]);
        return loss / static_cast<float>(n);
    }

    /**
     * @brief Computes the gradient of the MAE loss.
     * 
     * The derivative with respect to prediction \f$ \hat{y}_i \f$ is:
     * \f$ \frac{\partial L}{\partial \hat{y}_i} = \frac{1}{n} \cdot \text{sgn}(\hat{y}_i - y_i) \f$
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
            grad.data()[i] = factor * (diff > 0 ? 1.0f : (diff < 0 ? -1.0f : 0.0f));
        }
        return grad;
    }

    /**
     * @brief Returns the name of the loss function.
     */
    std::string getLossName() const override { return "Mean Absolute Error Loss"; }

    /**
     * @brief Creates a deep copy of the MAE loss function.
     * @return A shared pointer to the cloned object.
     */
    std::shared_ptr<owLoss> clone() const override { return std::make_shared<owMeanAbsoluteErrorLoss>(); }
};

} // namespace ow
