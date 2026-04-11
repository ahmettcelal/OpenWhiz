/*
 * owClippingLayer.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <algorithm>
#include <string>

namespace ow {

/**
 * @class owClippingLayer
 * @brief Performs element-wise clipping of input values to a specified range [min, max].
 * 
 * The owClippingLayer restricts input values to a safe range. It uses an internal
 * mask to track which values were clipped, allowing it to zero out gradients
 * for those elements during the backward pass.
 * 
 * Implementation Details:
 * - Forward: output = clamp(input, min, max).
 * - Mask: 1.0 if within range, 0.0 if clipped.
 * - Backward: gradient = gradient * mask.
 * 
 * Platform Notes:
 * - Computer/Mobile/Web: Lightweight and efficient for numerical stabilization.
 * - Industrial: Useful for data conditioning before passing to layers sensitive
 *   to large input magnitudes.
 * 
 * Comparison:
 * - Functional equivalent to owBoundingLayer, but implemented using an explicit
 *   mask tensor for the backward pass.
 */
class owClippingLayer : public owLayer {
public:
    /**
     * @brief Constructor for owClippingLayer.
     * @param min Minimum allowable value (default -1.0).
     * @param max Maximum allowable value (default 1.0).
     */
    owClippingLayer(float min = -1.0f, float max = 1.0f) : m_min(min), m_max(max) {
        m_layerName = "Clipping Layer";
    }

    /**
     * @brief Returns 0 as input size is dynamic.
     */
    size_t getInputSize() const override { return 0; }

    /**
     * @brief Returns 0 as output size is dynamic.
     */
    size_t getOutputSize() const override { return 0; }

    /**
     * @brief Dynamic layers ignore fixed neuron counts.
     */
    void setNeuronNum(size_t num) override { (void)num; }

    /**
     * @brief Creates a deep copy of the layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owClippingLayer>(m_min, m_max);
        copy->m_layerName = m_layerName;
        return copy;
    }

    /**
     * @brief Performs forward pass and generates the clipping mask.
     * @param input Input tensor.
     * @return Clipped tensor.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        m_mask = owTensor<float, 2>(input.shape());
        owTensor<float, 2> output(input.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            float val = input.data()[i];
            if (val < m_min) { output.data()[i] = m_min; m_mask.data()[i] = 0.0f; }
            else if (val > m_max) { output.data()[i] = m_max; m_mask.data()[i] = 0.0f; }
            else { output.data()[i] = val; m_mask.data()[i] = 1.0f; }
        }
        return output;
    }

    /**
     * @brief Performs backward pass using the stored mask.
     * @param outputGradient Gradient from the next layer.
     * @return Gradient with respect to the input.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        owTensor<float, 2> inputGradient = outputGradient;
        for (size_t i = 0; i < inputGradient.size(); ++i) inputGradient.data()[i] *= m_mask.data()[i];
        return inputGradient;
    }

    /**
     * @brief Serializes range to XML.
     */
    std::string toXML() const override {
        return "<Min>" + std::to_string(m_min) + "</Min><Max>" + std::to_string(m_max) + "</Max>";
    }

    /**
     * @brief Deserializes range from XML.
     */
    void fromXML(const std::string& xml) override {
        m_min = std::stof(getTagContent(xml, "Min")); m_max = std::stof(getTagContent(xml, "Max"));
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

private:
    float m_min, m_max; owTensor<float, 2> m_mask;
};

} // namespace ow
