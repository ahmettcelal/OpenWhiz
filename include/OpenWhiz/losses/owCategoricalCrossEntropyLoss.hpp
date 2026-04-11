/*
 * owCategoricalCrossEntropyLoss.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLoss.hpp"

namespace ow {

/**
 * @class owCategoricalCrossEntropyLoss
 * @brief Computes the Categorical Cross-Entropy loss for multi-class classification.
 * 
 * This loss function is used when the task involves more than two classes. 
 * It expects targets to be provided in a one-hot encoded format.
 * 
 * @details
 * Mathematical formula:
 * \f$ L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{i,j} \ln(\hat{y}_{i,j}) \f$
 * where \f$ N \f$ is the batch size, \f$ C \f$ is the number of classes, 
 * \f$ y_{i,j} \f$ is 1 if class \f$ j \f$ is the correct class for sample \f$ i \f$ 
 * and 0 otherwise, and \f$ \hat{y}_{i,j} \f$ is the predicted probability.
 * 
 * Features:
 * - Designed for multi-class classification.
 * - Works best when paired with a Softmax activation layer in the final output.
 * - Numerically stabilized using a small epsilon value.
 * 
 * Comparison with other losses:
 * - Use Categorical Cross-Entropy for multi-class classification.
 * - Use Binary Cross-Entropy when there are only two classes.
 * 
 * Platform notes:
 * - Computer: Standard for high-level classification models.
 * - Mobile/Web: Efficient as only the true class contribution is computed in the forward pass.
 * - Industrial: Reliable for complex multi-state machine monitoring.
 */
class owCategoricalCrossEntropyLoss : public owLoss {
public:
    /**
     * @brief Computes the Categorical Cross-Entropy loss value.
     * @param prediction Predicted probabilities for each class.
     * @param target One-hot encoded ground truth labels.
     * @return Computed loss value.
     */
    float compute(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        float loss = 0.0f;
        size_t batchSize = prediction.shape()[0];
        const float eps = 1e-12f;
        for (size_t i = 0; i < prediction.size(); ++i) {
            if (target.data()[i] > 0.5f) { // One-hot target
                float p = std::max(eps, std::min(1.0f - eps, prediction.data()[i]));
                loss -= std::log(p);
            }
        }
        return loss / static_cast<float>(batchSize);
    }

    /**
     * @brief Computes the gradient of the Categorical Cross-Entropy loss.
     * 
     * Derivative with respect to \f$ \hat{y}_{i,j} \f$:
     * \f$ \frac{\partial L}{\partial \hat{y}_{i,j}} = -\frac{y_{i,j}}{\hat{y}_{i,j} \cdot N} \f$
     * 
     * @param prediction Predicted probabilities.
     * @param target One-hot encoded labels.
     * @return Gradient tensor.
     */
    owTensor<float, 2> gradient(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        owTensor<float, 2> grad(prediction.shape());
        size_t batchSize = prediction.shape()[0];
        const float eps = 1e-12f;
        for (size_t i = 0; i < prediction.size(); ++i) {
            float p = std::max(eps, std::min(1.0f - eps, prediction.data()[i]));
            grad.data()[i] = -target.data()[i] / (p * static_cast<float>(batchSize));
        }
        return grad;
    }

    /**
     * @brief Returns the name of the loss function.
     */
    std::string getLossName() const override { return "Categorical Cross-Entropy Loss"; }

    /**
     * @brief Creates a deep copy of the Categorical Cross-Entropy loss function.
     * @return A shared pointer to the cloned object.
     */
    std::shared_ptr<owLoss> clone() const override { return std::make_shared<owCategoricalCrossEntropyLoss>(); }
};

} // namespace ow
