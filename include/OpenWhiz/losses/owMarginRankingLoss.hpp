/*
 * owMarginRankingLoss.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLoss.hpp"

namespace ow {

/**
 * @class owMarginRankingLoss
 * @brief Computes the loss for ranking pairs of inputs.
 * 
 * Margin Ranking Loss is used when the model needs to learn the relative 
 * ordering between pairs of items rather than their absolute values.
 * 
 * @details
 * Mathematical formula:
 * \f$ L(x_1, x_2, y) = \max(0, -y(x_1 - x_2) + \text{margin}) \f$
 * where:
 * - \f$ x_1, x_2 \f$ are the scores of the two items being compared.
 * - \f$ y \f$ is the target label: 1 if \f$ x_1 \f$ should be ranked higher than 
 *   \f$ x_2 \f$, and -1 otherwise.
 * - \f$ \text{margin} \f$ is the minimum desired gap between the scores.
 * 
 * Features:
 * - Effectively handles relative ordering tasks.
 * - Robust to the absolute scale of the predicted scores.
 * - Only penalizes the model when the margin constraint is violated.
 * 
 * Comparison with other losses:
 * - Use Margin Ranking Loss for recommendation systems or search ranking.
 * - Use Binary Cross-Entropy if you are performing independent classification 
 *   of items.
 * 
 * Platform notes:
 * - Computer: Frequently used in large-scale information retrieval.
 * - Mobile/Web: Suitable for personalized ranking on devices.
 * - Industrial: Can be used for prioritizing maintenance tasks or alerts.
 */
class owMarginRankingLoss : public owLoss {
public:
    /**
     * @brief Constructs Margin Ranking Loss with a specific margin.
     * @param margin The minimum desired distance between the pair of scores.
     */
    owMarginRankingLoss(float margin = 1.0f) : m_margin(margin) {}

    /**
     * @brief Computes the Margin Ranking Loss value.
     * 
     * Expects a prediction tensor with 2 columns representing (x1, x2).
     * 
     * @param prediction Tensor of shape [BatchSize, 2].
     * @param target Tensor of shape [BatchSize, 1] containing labels (1 or -1).
     * @return Average ranking loss.
     * @throws std::runtime_error If prediction shape is incorrect.
     */
    float compute(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        if (prediction.shape()[1] != 2) throw std::runtime_error("MarginRankingLoss expects prediction with 2 columns (x1, x2)");
        float loss = 0.0f;
        size_t n = prediction.shape()[0];
        for (size_t i = 0; i < n; ++i) {
            float x1 = prediction(i, 0);
            float x2 = prediction(i, 1);
            float y = target(i, 0);
            float val = -y * (x1 - x2) + m_margin;
            if (val > 0) loss += val;
        }
        return loss / static_cast<float>(n);
    }

    /**
     * @brief Computes the gradient of the Margin Ranking Loss.
     * 
     * @param prediction Predicted scores.
     * @param target Ground truth ranking labels.
     * @return Gradient tensor of shape [BatchSize, 2].
     */
    owTensor<float, 2> gradient(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        owTensor<float, 2> grad(prediction.shape());
        size_t n = prediction.shape()[0];
        float factor = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            float x1 = prediction(i, 0);
            float x2 = prediction(i, 1);
            float y = target(i, 0);
            if (-y * (x1 - x2) + m_margin > 0) {
                grad(i, 0) = -y * factor;
                grad(i, 1) = y * factor;
            } else {
                grad(i, 0) = 0;
                grad(i, 1) = 0;
            }
        }
        return grad;
    }

    /**
     * @brief Returns the name of the loss function.
     */
    std::string getLossName() const override { return "Margin Ranking Loss"; }

    /**
     * @brief Creates a deep copy of the Margin Ranking Loss function.
     * @return A shared pointer to the cloned object.
     */
    std::shared_ptr<owLoss> clone() const override { return std::make_shared<owMarginRankingLoss>(m_margin); }
private:
    float m_margin;
};

} // namespace ow
