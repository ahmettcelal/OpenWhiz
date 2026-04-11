/*
 * owBoundingLayer.hpp
 *
 *  Created on: Jan 12, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <algorithm>

namespace ow {

/**
 * @class owBoundingLayer
 * @brief Clips input values to a specified [min, max] range.
 * 
 * The owBoundingLayer ensures that all output values fall within a defined boundary.
 * Unlike a simple activation, it also zeros out gradients for values that were
 * outside the bounds during the forward pass, effectively stopping learning for
 * saturated inputs.
 * 
 * Implementation Details:
 * - Forward: output = clamp(input, min, max).
 * - Backward: gradient = (input >= min && input <= max) ? gradient : 0.
 * 
 * Platform Notes:
 * - Industrial: Critical for safety-critical systems to ensure control signals 
 *   do not exceed physical actuator limits.
 * - Computer/Mobile: Useful for preventing numerical instability (NaN/Inf) in deep networks.
 * 
 * Comparison:
 * - Similar to owClippingLayer, but specifically designed to "freeze" gradients 
 *   outside the valid range.
 */
class owBoundingLayer : public owLayer {
public:
    /**
     * @brief Constructor for owBoundingLayer.
     * @param min Minimum allowable value.
     * @param max Maximum allowable value.
     */
    owBoundingLayer(float min = 0.0f, float max = 1.0f) : m_min(min), m_max(max) {
        m_layerName = "Bounding Layer";
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
    void setNeuronNum(size_t num) override {}

    /**
     * @brief Updates the boundary range.
     * @param min New minimum.
     * @param max New maximum.
     */
    void setBounds(float min, float max) { m_min = min; m_max = max; }

    /**
     * @brief Creates a deep copy of the layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owBoundingLayer>(m_min, m_max);
        copy->m_layerName = m_layerName;
        return copy;
    }

    /**
     * @brief Serializes bounds to XML.
     */
    std::string toXML() const override {
        std::stringstream ss; ss << "<Min>" << m_min << "</Min>\n<Max>" << m_max << "</Max>\n";
        return ss.str();
    }

    /**
     * @brief Deserializes bounds from XML.
     */
    void fromXML(const std::string& xml) override {
        m_min = std::stof(getTagContent(xml, "Min"));
        m_max = std::stof(getTagContent(xml, "Max"));
    }

    /**
     * @brief Performs forward pass: clips values to [min, max].
     * @param input Input tensor.
     * @return Bounded tensor.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        m_lastInput = input;
        owTensor<float, 2> output = input;
        for (size_t i = 0; i < output.size(); ++i) {
            output.data()[i] = std::max(m_min, std::min(m_max, output.data()[i]));
        }
        return output;
    }

    /**
     * @brief Performs backward pass: zeros gradients for out-of-bounds elements.
     * @param outputGradient Gradient from the next layer.
     * @return Masked gradient.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        owTensor<float, 2> inputGradient = outputGradient;
        for (size_t i = 0; i < inputGradient.size(); ++i) {
            if (m_lastInput.data()[i] < m_min || m_lastInput.data()[i] > m_max) inputGradient.data()[i] = 0.0f;
        }
        return inputGradient;
    }

    /**
     * @brief Training step (no-op).
     */
    void train() override {}

	float* getParamsPtr() override { return nullptr; }
	float* getGradsPtr() override { return nullptr; }
	size_t getParamsCount() override { return 0; }

private:
    float m_min, m_max; owTensor<float, 2> m_lastInput;
};

} // namespace ow
