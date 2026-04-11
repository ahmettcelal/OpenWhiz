/*
 * owDateTimeEncodingLayer.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <cmath>
#include <stdexcept>

namespace ow {

/**
 * @class owDateTimeEncodingLayer
 * @brief Encodes temporal features into cyclic (sin/cos) representations.
 * 
 * The owDateTimeEncodingLayer transforms discrete time components (Hour, Day of Week,
 * Month, Day of Month) into a continuous cyclic space. This preserves temporal
 * continuity (e.g., ensuring that 23:59 is seen as close to 00:01).
 * 
 * Input Format:
 * - Expects 4 features: [Hour(0-23), DayOfWeek(0-6), Month(1-12), DayOfMonth(1-31)].
 * 
 * Output Format:
 * - Produces 8 features: [sin(H), cos(H), sin(DoW), cos(DoW), sin(M), cos(M), sin(DoM), cos(DoM)].
 * 
 * Platform Notes:
 * - Industrial/IoT: Essential for time-series forecasting, energy load prediction,
 *   and scheduled maintenance models.
 * - Computer/Mobile: Lightweight preprocessing that significantly improves 
 *   model convergence on temporal data.
 */
class owDateTimeEncodingLayer : public owLayer {
public:
    /**
     * @brief Constructor for owDateTimeEncodingLayer.
     */
    owDateTimeEncodingLayer() {
        m_layerName = "DateTime Encoding Layer";
    }

    /**
     * @brief Returns 4 as the required input feature count.
     */
    size_t getInputSize() const override { return 4; }

    /**
     * @brief Returns 8 as the produced output feature count.
     */
    size_t getOutputSize() const override { return 8; }

    /**
     * @brief Fixed encoding layers ignore neuron resizing.
     */
    void setNeuronNum(size_t num) override { (void)num; }

    /**
     * @brief Creates a deep copy of the layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owDateTimeEncodingLayer>();
        copy->m_layerName = m_layerName;
        return copy;
    }

    /**
     * @brief Serializes layer info to XML.
     */
    std::string toXML() const override { return "<Info>Cyclic Encoding</Info>\n"; }

    /**
     * @brief Deserializes layer info from XML.
     */
    void fromXML(const std::string& xml) override { (void)xml; }

    /**
     * @brief Performs forward pass: maps 4 features to 8 cyclic features.
     * @param input Input tensor of shape [Batch, 4].
     * @return Output tensor of shape [Batch, 8].
     * @throws std::runtime_error if input second dimension is not 4.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        if (input.shape()[1] != 4) throw std::runtime_error("DateTimeEncodingLayer expects 4 features (H, DoW, M, DoM)");
        owTensor<float, 2> output(input.shape()[0], 8);
        
        for (size_t b = 0; b < input.shape()[0]; ++b) {
            // Hour (24)
            output(b, 0) = std::sin(2.0f * OW_PI * input(b, 0) / 24.0f);
            output(b, 1) = std::cos(2.0f * OW_PI * input(b, 0) / 24.0f);
            // Day of Week (7)
            output(b, 2) = std::sin(2.0f * OW_PI * input(b, 1) / 7.0f);
            output(b, 3) = std::cos(2.0f * OW_PI * input(b, 1) / 7.0f);
            // Month (12)
            output(b, 4) = std::sin(2.0f * OW_PI * (input(b, 2) - 1.0f) / 12.0f);
            output(b, 5) = std::cos(2.0f * OW_PI * (input(b, 2) - 1.0f) / 12.0f);
            // Day of Month (31)
            output(b, 6) = std::sin(2.0f * OW_PI * (input(b, 3) - 1.0f) / 31.0f);
            output(b, 7) = std::cos(2.0f * OW_PI * (input(b, 3) - 1.0f) / 31.0f);
        }
        return output;
    }

    /**
     * @brief Performs backward pass. Since encoding is fixed, it returns zero gradients.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        owTensor<float, 2> inputGradient(outputGradient.shape()[0], 4);
        inputGradient.setZero();
        return inputGradient;
    }

    /**
     * @brief Training step (no-op).
     */
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
