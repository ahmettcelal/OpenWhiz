/*
 * owAffineLayer.hpp
 *
 *  Created on: Jan 12, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <sstream>

namespace ow {

/**
 * @class owAffineLayer
 * @brief Performs a global scalar affine transformation: y = ax + b.
 * 
 * The owAffineLayer applies a single learnable scale factor (a) and a single
 * learnable shift (b) to all elements of the input tensor. 
 * 
 * Implementation Details:
 * - Operates element-wise: output[i] = input[i] * a + b.
 * - Both 'a' and 'b' are scalars applied globally to the entire tensor.
 * - Parameters are learnable through backpropagation if an optimizer is attached.
 * 
 * Distinction:
 * - Unlike owLinearLayer, which performs matrix multiplication (fully connected),
 *   owAffineLayer is a simple scaling and shifting operation.
 * - It is much lighter than owLinearLayer as it only has 2 learnable parameters.
 * 
 * Platform Notes:
 * - Computer/Mobile/Web: Very low overhead, ideal for final output scaling.
 * - Industrial: Useful for engineering unit conversion (e.g., Voltage to Temperature).
 */
class owAffineLayer : public owLayer {
public:
    /**
     * @brief Constructor for owAffineLayer.
     * Initializes a=1.0 and b=0.0 (identity transformation).
     */
    owAffineLayer() : m_a(1, 1), m_b(1, 1), m_aGrad(1, 1), m_bGrad(1, 1) {
        m_layerName = "Affine Layer";
        m_a(0, 0) = 1.0f;
        m_b(0, 0) = 0.0f;
    }

    /**
     * @brief Returns 0 as input size is dynamic.
     */
    size_t getInputSize() const override { return 0; }

    /**
     * @brief Returns 0 as output size is dynamic (same as input).
     */
    size_t getOutputSize() const override { return 0; }

    /**
     * @brief Dynamic layers ignore fixed neuron counts.
     */
    void setNeuronNum(size_t num) override { (void)num; }

    /**
     * @brief Returns the current scale factor 'a'.
     */
    float getA() const { return m_a(0, 0); }

    /**
     * @brief Returns the current shift factor 'b'.
     */
    float getB() const { return m_b(0, 0); }

    /**
     * @brief Creates a deep copy of the layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owAffineLayer>();
        copy->m_a = m_a; copy->m_b = m_b; copy->m_layerName = m_layerName;
        return copy;
    }

    /**
     * @brief Serializes 'a' and 'b' to XML.
     */
    std::string toXML() const override {
        std::stringstream ss;
        ss << "<A>" << m_a(0, 0) << "</A>\n<B>" << m_b(0, 0) << "</B>\n";
        return ss.str();
    }

    /**
     * @brief Deserializes 'a' and 'b' from XML.
     */
    void fromXML(const std::string& xml) override {
        m_a(0, 0) = std::stof(getTagContent(xml, "A"));
        m_b(0, 0) = std::stof(getTagContent(xml, "B"));
    }

    /**
     * @brief Performs forward pass: output = input * a + b.
     * @param input Input tensor of any shape.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        m_lastInput = input;
        float a = m_a(0, 0);
        float b = m_b(0, 0);
        owTensor<float, 2> output = input;
        for (size_t i = 0; i < output.size(); ++i) {
            output.data()[i] = output.data()[i] * a + b;
        }
        return output;
    }

    /**
     * @brief Performs backward pass, computing gradients for 'a' and 'b'.
     * @param outputGradient Gradient from the next layer.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        float a = m_a(0, 0);
        owTensor<float, 2> inputGradient = outputGradient;
        
        m_aGrad.setZero();
        m_bGrad.setZero();

        for (size_t i = 0; i < outputGradient.size(); ++i) {
            m_aGrad(0, 0) += outputGradient.data()[i] * m_lastInput.data()[i];
            m_bGrad(0, 0) += outputGradient.data()[i];
            inputGradient.data()[i] *= a;
        }
        return inputGradient;
    }

    /**
     * @brief Updates 'a' and 'b' using the attached optimizer.
     */
    void train() override {
        if (m_optimizer) {
            m_optimizer->update(m_a, m_aGrad);
            m_optimizer->update(m_b, m_bGrad);
        }
    }

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
    owTensor<float, 2> m_a, m_b, m_aGrad, m_bGrad, m_lastInput;
};

} // namespace ow
