/*
 * owRescaling.hpp
 *
 *  Created on: Jan 12, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"

namespace ow {

/**
 * @class owRescalingLayer
 * @brief Performs a linear transformation (y = ax + b) on the input tensor.
 * 
 * This layer applies a constant scaling factor (a) and a constant offset (b) to all 
 * elements of the input tensor. It is often used for simple data preprocessing or 
 * adjusting the range of outputs (e.g., from [0, 1] to [-1, 1]).
 * 
 * @details
 * **Implementation Details:**
 * - Element-wise operation: output = input * m_a + m_b.
 * - Non-trainable layer (a and b are constant once set).
 * - Forward pass scales and offsets; backward pass scales the gradient by `a`.
 * 
 * **Unique Features:**
 * - Extremely low computational overhead.
 * - Simple way to shift or scale inputs/outputs without learned parameters.
 * 
 * **Platform-Specific Notes:**
 * - **Computer/Mobile/Web:** Performance is very high; suitable for real-time preprocessing.
 * - **Industrial:** Can be used to convert raw sensor voltages to meaningful physical units 
 *   using linear calibration constants.
 */
class owRescalingLayer : public owLayer {
public:
    /**
     * @brief Constructs a Rescaling layer.
     * @param a The scaling factor (slope). Defaults to 1.0.
     * @param b The offset value (intercept). Defaults to 0.0.
     */
    owRescalingLayer(float a = 1.0f, float b = 0.0f) : m_a(a), m_b(b) {
        m_layerName = "Rescaling Layer";
    }
    size_t getInputSize() const override { return 0; } // Dynamic
    size_t getOutputSize() const override { return 0; }
    void setNeuronNum(size_t num) override {}

    void setA(float a) { m_a = a; }
    void setB(float b) { m_b = b; }
    float getA() const { return m_a; }
    float getB() const { return m_b; }
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owRescalingLayer>(m_a, m_b);
        copy->m_layerName = m_layerName;
        return copy;
    }

    std::string toXML() const override {
        std::stringstream ss;
        ss << "<A>" << m_a << "</A>\n<B>" << m_b << "</B>\n";
        return ss.str();
    }
    void fromXML(const std::string& xml) override {
        m_a = std::stof(getTagContent(xml, "A"));
        m_b = std::stof(getTagContent(xml, "B"));
    }

    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        owTensor<float, 2> output = input;
        for (size_t i = 0; i < output.size(); ++i) {
            output.data()[i] = output.data()[i] * m_a + m_b;
        }
        return output;
    }
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        owTensor<float, 2> inputGradient = outputGradient;
        for (size_t i = 0; i < inputGradient.size(); ++i) {
            inputGradient.data()[i] *= m_a;
        }
        return inputGradient;
    }
    void train() override {}

	float* getParamsPtr() override { return nullptr; }
	float* getGradsPtr() override { return nullptr; }
	size_t getParamsCount() override { return 0; }

private:
    float m_a, m_b;
};

} // namespace ow
