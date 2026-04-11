/*
 * owBinaryCrossEntropyLoss.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLoss.hpp"

namespace ow {

/**
 * @class owBinaryCrossEntropyLoss
 * @brief Computes the Binary Cross-Entropy (BCE) loss, also known as Log Loss.
 * 
 * BCE loss measures the performance of a classification model whose output is a 
 * probability value between 0 and 1.
 * 
 * @details
 * Mathematical formula:
 * \f$ L = -\frac{1}{n} \sum_{i=1}^{n} [y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i)] \f$
 * where \f$ y_i \f$ is the target label (0 or 1) and \f$ \hat{y}_i \f$ is the 
 * predicted probability.
 * 
 * Features:
 * - Ideally suited for binary classification tasks.
 * - Heavily penalizes predictions that are confident but wrong.
 * - Employs a small epsilon value to prevent mathematical errors like \f$ \ln(0) \f$.
 * 
 * Comparison with other losses:
 * - Use BCE for binary classification.
 * - Use Categorical Cross-Entropy for multi-class classification.
 * - Avoid using MSE for classification as it leads to slower convergence due 
 *   to the "flatness" of the sigmoid derivative at extremes.
 * 
 * Platform notes:
 * - Computer/Mobile/Web: Standard for neural network classifiers.
 * - Industrial: Robust for binary diagnostic tools (e.g., pass/fail detection).
 */
class owBinaryCrossEntropyLoss : public owLoss {
public:
    /**
     * @brief Computes the BCE loss value.
     * @param prediction Predicted probabilities (should be in range [0, 1]).
     * @param target Ground truth labels (0 or 1).
     * @return Computed Log Loss.
     */
    float compute(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        float loss = 0.0f;
        size_t n = prediction.size();
        const float eps = 1e-12f; // Numerical stability
        for (size_t i = 0; i < n; ++i) {
            float p = std::max(eps, std::min(1.0f - eps, prediction.data()[i]));
            float t = target.data()[i];
            loss -= (t * std::log(p) + (1.0f - t) * std::log(1.0f - p));
        }
        return loss / static_cast<float>(n);
    }

    /**
     * @brief Computes the gradient of the BCE loss.
     * 
     * Derivative with respect to prediction \f$ \hat{y}_i \f$:
     * \f$ \frac{\partial L}{\partial \hat{y}_i} = \frac{1}{n} [ -\frac{y_i}{\hat{y}_i} + \frac{1 - y_i}{1 - \hat{y}_i} ] \f$
     * 
     * @param prediction Predicted probabilities.
     * @param target Ground truth labels.
     * @return Gradient tensor.
     */
    owTensor<float, 2> gradient(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        owTensor<float, 2> grad(prediction.shape());
        size_t n = prediction.size();
        const float eps = 1e-12f;
        float factor = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            float p = std::max(eps, std::min(1.0f - eps, prediction.data()[i]));
            float t = target.data()[i];
            grad.data()[i] = factor * (-(t / p) + (1.0f - t) / (1.0f - p));
        }
        return grad;
    }

    /**
     * @brief Returns the name of the loss function.
     */
    std::string getLossName() const override { return "Binary Cross-Entropy Loss"; }

    /**
     * @brief Creates a deep copy of the BCE loss function.
     * @return A shared pointer to the cloned object.
     */
    std::shared_ptr<owLoss> clone() const override { return std::make_shared<owBinaryCrossEntropyLoss>(); }
};

} // namespace ow
