/*
 * owAdditionLayer.hpp
 *
 *  Created on: Jan 12, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"

namespace ow {

/**
 * @class owAdditionLayer
 * @brief A layer that adds a constant vector to the input tensor.
 * 
 * The owAdditionLayer performs an element-wise addition of a stored vector (values)
 * to every sample in the input batch. It is useful for applying a fixed bias or
 * shifting the data distribution.
 * 
 * Implementation Details:
 * - Performs Y = X + V where V is the stored values vector.
 * - The vector V has the same size as the input feature dimension.
 * 
 * Platform Notes:
 * - Computer/Mobile/Web: Highly efficient due to simple element-wise addition.
 * - Industrial: Useful for sensor offset calibration or baseline adjustment.
 * 
 * Comparison:
 * - Unlike owLinearLayer, this does not involve matrix multiplication.
 * - Unlike owAffineLayer, this uses a vector of values instead of a single scalar.
 */
class owAdditionLayer : public owLayer {
public:
    /**
     * @brief Constructor for owAdditionLayer.
     * @param size The number of features in the input and output.
     */
    owAdditionLayer(size_t size) : m_size(size), m_values(1, size) {
        m_layerName = "Addition Layer";
        m_values.setZero();
    }

    /**
     * @brief Returns the expected input feature size.
     */
    size_t getInputSize() const override { return m_size; }

    /**
     * @brief Returns the output feature size.
     */
    size_t getOutputSize() const override { return m_size; }

    /**
     * @brief Sets the number of neurons (features) and reinitializes values to zero.
     * @param num New feature size.
     */
    void setNeuronNum(size_t num) override {
        m_size = num; m_values = owTensor<float, 2>(1, m_size); m_values.setZero();
    }

    /**
     * @brief Manually sets the values to be added.
     * @param vals A 1xN tensor containing the addition values.
     */
    void setValues(const owTensor<float, 2>& vals) { m_values = vals; }

    /**
     * @brief Creates a deep copy of the layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owAdditionLayer>(m_size);
        copy->m_values = m_values; copy->m_layerName = m_layerName;
        return copy;
    }

    /**
     * @brief Serializes the layer configuration to XML.
     */
    std::string toXML() const override {
        std::stringstream ss; ss << "<Size>" << m_size << "</Size>\n<Values>" << m_values.toString() << "</Values>\n";
        return ss.str();
    }

    /**
     * @brief Deserializes the layer configuration from XML.
     */
    void fromXML(const std::string& xml) override {
        m_size = std::stoul(getTagContent(xml, "Size"));
        m_values = owTensor<float, 2>(1, m_size);
        m_values.fromString(getTagContent(xml, "Values"));
    }

    /**
     * @brief Performs the forward pass: output = input + values.
     * @param input Input tensor of shape [BatchSize, Size].
     * @return Resulting tensor after addition.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        owTensor<float, 2> output = input;
        for (size_t b = 0; b < input.shape()[0]; ++b) {
            for (size_t f = 0; f < m_size; ++f) output(b, f) += m_values(0, f);
        }
        return output;
    }

    /**
     * @brief Performs the backward pass. Addition layer passes the gradient unchanged.
     * @param outputGradient Gradient of the loss with respect to the output.
     * @return Gradient with respect to the input.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        return outputGradient; 
    }

    /**
     * @brief Training step. Addition layer has no trainable parameters in this implementation.
     */
    void train() override {}

	float* getParamsPtr() override { return nullptr; }
	float* getGradsPtr() override { return nullptr; }
	size_t getParamsCount() override { return 0; }

private:
    size_t m_size; owTensor<float, 2> m_values;
};

} // namespace ow
