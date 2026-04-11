/*
 * owSmoothingLayer.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <string>

namespace ow {

/**
 * @class owSmoothingLayer
 * @brief Applies Exponential Smoothing to reduce noise in time-series data.
 * 
 * This layer implements Simple Exponential Smoothing (SES), where the output is a 
 * weighted average of the current input and the previous smoothed value.
 * 
 * @details
 * **Implementation Details:**
 * - Formula: S_t = alpha * X_t + (1 - alpha) * S_{t-1}.
 * - `alpha` controls the level of smoothing (0 < alpha < 1). 
 * - Values of `alpha` close to 1 give more weight to recent data, while values close 
 *   to 0 give more weight to historical data.
 * - This is an online smoothing process that operates along the batch dimension 
 *   (interpreted as time).
 * 
 * **Unique Features:**
 * - Effective for noise reduction without introducing significant latency.
 * - Simple parameterization with a single `alpha` value.
 * 
 * **Platform-Specific Notes:**
 * - **Computer/Mobile/Web:** Negligible performance cost.
 * - **Industrial:** Crucial for filtering noisy sensor data before feeding it into 
 *   predictive models or control logic.
 */
class owSmoothingLayer : public owLayer {
public:
    /**
     * @brief Constructs a Smoothing layer.
     * @param alpha The smoothing factor (0 to 1). Defaults to 0.5.
     */
	owSmoothingLayer(float alpha = 0.5f) : m_alpha(alpha) {
		m_layerName = "Smoothing Layer";
	}

	owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
		owTensor<float, 2> output(input.shape());
		size_t batch = input.shape()[0];
		size_t features = input.shape()[1];

		for (size_t f = 0; f < features; ++f) {
			float lastS = input(0, f);
			output(0, f) = lastS;
			for (size_t b = 1; b < batch; ++b) {
				float currentS = m_alpha * input(b, f) + (1.0f - m_alpha) * lastS;
				output(b, f) = currentS;
				lastS = currentS;
			}
		}
		return output;
	}

	owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
		return outputGradient;
	}

	std::shared_ptr<owLayer> clone() const override {
		auto copy = std::make_shared<owSmoothingLayer>(m_alpha);
		copy->m_layerName = m_layerName;
		return copy;
	}

	std::string toXML() const override { return "<Alpha>" + std::to_string(m_alpha) + "</Alpha>"; }
	void fromXML(const std::string& xml) override { m_alpha = std::stof(getTagContent(xml, "Alpha")); }

	void train() override {}
	size_t getInputSize() const override { return 0; }
	size_t getOutputSize() const override { return 0; }
	void setNeuronNum(size_t num) override { (void)num; }

	float* getParamsPtr() override {
		return nullptr;
	}

	float* getGradsPtr() override {
		return nullptr;
	}

	size_t getParamsCount() override {
		return 0;
	}


private:
	float m_alpha;
};

} // namespace ow
