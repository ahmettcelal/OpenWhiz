/*
 * owQuantileLayer.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"

namespace ow {

/**
 * @class owQuantileLayer
 * @brief Acts as a placeholder or marker for Quantile Regression outputs.
 * 
 * This layer is specifically designed for models that perform Quantile Regression, where 
 * the network predicts specific percentiles (quantiles) of the target distribution rather 
 * than just the mean.
 * 
 * @details
 * **Implementation Details:**
 * - Currently implements an identity operation (forward and backward passes return the input).
 * - Serves as a semantic marker to indicate that the preceding layer's output should be 
 *   interpreted as multiple quantiles.
 * 
 * **Unique Features:**
 * - Used in conjunction with `owPinballLoss` to implement probabilistic forecasting.
 * - Allows a single network to output multiple prediction intervals (e.g., 10th, 50th, 90th 
 *   percentiles).
 * 
 * **Platform-Specific Notes:**
 * - **Computer/Mobile/Web:** No computational overhead.
 * - **Industrial:** Essential for risk management in industrial processes, providing not just 
 *   a prediction but an estimate of uncertainty (confidence intervals).
 */
class owQuantileLayer : public owLayer {
public:
    /**
     * @brief Constructs a Quantile marker layer.
     */
    owQuantileLayer() { m_layerName = "Quantile Layer"; }
    size_t getInputSize() const override { return 0; }
    size_t getOutputSize() const override { return 0; }
    void setNeuronNum(size_t num) override { (void)num; }
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owQuantileLayer>();
        copy->m_layerName = m_layerName;
        return copy;
    }

    std::string toXML() const override { return "<Info>Quantile</Info>"; }
    void fromXML(const std::string& xml) override { (void)xml; }

    owTensor<float, 2> forward(const owTensor<float, 2>& input) override { return input; }
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override { return outputGradient; }
    void train() override {}

	float* getParamsPtr() override {
		return nullptr;
	}

	float* getGradsPtr() override {
		return nullptr;
	}

	size_t getParamsCount() override {
		return 0;
	}

};

} // namespace ow
