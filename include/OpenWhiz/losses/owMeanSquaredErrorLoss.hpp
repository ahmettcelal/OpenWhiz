/*
 * owMeanSquaredErrorLoss.hpp
 *
 *  Created on: Dec 16, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLoss.hpp"

namespace ow {

/**
 * @class owMeanSquaredErrorLoss
 * @brief Computes the Mean Squared Error (MSE) loss.
 * 
 * MSE loss calculates the average of the squares of the differences between 
 * predictions and targets.
 * 
 * @details
 * Mathematical formula:
 * \f$ L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \f$
 * where \f$ y_i \f$ is the target and \f$ \hat{y}_i \f$ is the prediction.
 * 
 * Features:
 * - Penalizes outliers more heavily than MAE due to the squaring operation.
 * - Resulting loss is always non-negative.
 * - Smooth and differentiable everywhere, which aids convergence in optimization.
 * 
 * Comparison with other losses:
 * - Use MSE when you want to heavily penalize large errors (outliers).
 * - Use MAE (Mean Absolute Error) when your data contains many outliers that should not dominate the loss.
 * - Use Huber loss as a robust alternative that behaves like MSE for small errors and MAE for large ones.
 * 
 * Platform notes:
 * - Computer: Standard choice for most regression tasks.
 * - Mobile/Web: Efficient to compute due to simple arithmetic operations.
 * - Industrial: Good for high-precision tasks where small deviations need to be minimized.
 */
class owMeanSquaredErrorLoss : public owLoss {
public:
    /**
     * @brief Computes the MSE loss value.
     * @param prediction Predicted output tensor.
     * @param target Ground truth tensor.
     * @return The average squared difference.
     */
    float compute(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        float loss = 0.0f;
        size_t n = prediction.size();
        for (size_t i = 0; i < n; ++i) {
            float diff = prediction.data()[i] - target.data()[i];
            loss += diff * diff;
        }
        return loss / static_cast<float>(n);
    }

    /**
     * @brief Computes the gradient of the MSE loss.
     * 
     * The derivative with respect to prediction \f$ \hat{y}_i \f$ is:
     * \f$ \frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n} (\hat{y}_i - y_i) \f$
     * 
     * @param prediction Predicted output tensor.
     * @param target Ground truth tensor.
     * @return Gradient tensor of the same shape as prediction.
     */
    owTensor<float, 2> gradient(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        owTensor<float, 2> grad(prediction.shape());
        size_t n = prediction.size();
        float factor = 2.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) grad.data()[i] = factor * (prediction.data()[i] - target.data()[i]);
        return grad;
    }

    /**
     * @brief Returns the name of the loss function.
     */
    std::string getLossName() const override { return "Mean Squared Error Loss"; }

    /**
     * @brief Creates a deep copy of the MSE loss function.
     * @return A shared pointer to the cloned object.
     */
    std::shared_ptr<owLoss> clone() const override { return std::make_shared<owMeanSquaredErrorLoss>(); }
};

} // namespace ow
