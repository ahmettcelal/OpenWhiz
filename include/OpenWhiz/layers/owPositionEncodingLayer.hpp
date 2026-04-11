/*
 * owPositionEncodingLayer.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <vector>
#include <cmath>
#include <sstream>

namespace ow {

/**
 * @class owPositionEncodingLayer
 * @brief Injects positional information into sequence embeddings for Transformer models.
 * 
 * Since attention-based models (like Multi-Head Attention) don't have an inherent sense of 
 * sequence order, this layer adds sinusoidal positional encodings to the input.
 * 
 * @details
 * **Implementation Details:**
 * - Uses fixed sine and cosine functions of different frequencies.
 * - Formula: PE(pos, 2i) = sin(pos / 10000^(2i/dModel)), PE(pos, 2i+1) = cos(pos / 10000^(2i/dModel)).
 * - Pre-calculates the encoding matrix up to `maxLen` to optimize forward pass performance.
 * 
 * **Unique Features:**
 * - Constant positional encodings that don't add trainable parameters.
 * - Allows the model to generalize to sequence lengths slightly longer than those seen during training.
 * 
 * **Platform-Specific Notes:**
 * - **Computer:** Very fast; encoding is calculated once and added to input batches.
 * - **Mobile/Web:** Low memory footprint as the encoding table is small.
 * - **Industrial:** Crucial for time-series forecasting where the relative time position of a sensor 
 *   reading is significant.
 */
class owPositionEncodingLayer : public owLayer {
public:
    /**
     * @brief Constructs a Position Encoding layer.
     * @param maxLen The maximum sequence length supported by the pre-calculated table.
     * @param dModel The dimensionality of the input embeddings.
     */
    owPositionEncodingLayer(size_t maxLen, size_t dModel) : m_maxLen(maxLen), m_dModel(dModel), m_encoding(maxLen, dModel) {
        m_layerName = "Position Encoding Layer";
        for (size_t pos = 0; pos < maxLen; ++pos) {
            for (size_t i = 0; i < dModel; i += 2) {
                float divTerm = std::pow(10000.0f, (float)i / (float)dModel);
                m_encoding(pos, i) = std::sin((float)pos / divTerm);
                if (i + 1 < dModel) m_encoding(pos, i + 1) = std::cos((float)pos / divTerm);
            }
        }
    }

    size_t getInputSize() const override { return m_dModel; }
    size_t getOutputSize() const override { return m_dModel; }
    void setNeuronNum(size_t num) override { (void)num; }
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owPositionEncodingLayer>(m_maxLen, m_dModel);
        copy->m_layerName = m_layerName;
        return copy;
    }

    std::string toXML() const override {
        std::stringstream ss; ss << "<MaxLen>" << m_maxLen << "</MaxLen>\n<DModel>" << m_dModel << "</DModel>\n";
        return ss.str();
    }
    void fromXML(const std::string& xml) override {
        m_maxLen = std::stoul(getTagContent(xml, "MaxLen"));
        m_dModel = std::stoul(getTagContent(xml, "DModel"));
    }

    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        owTensor<float, 2> output = input;
        size_t batchSize = input.shape()[0];
        for (size_t b = 0; b < batchSize; ++b) {
            size_t pos = b % m_maxLen; // Simple assumption: batch maps to sequence positions
            for (size_t i = 0; i < m_dModel; ++i) output(b, i) += m_encoding(pos, i);
        }
        return output;
    }
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        return outputGradient; // Constant addition, gradient is unchanged
    }
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

private:
    size_t m_maxLen, m_dModel;
    owTensor<float, 2> m_encoding;
};

} // namespace ow
