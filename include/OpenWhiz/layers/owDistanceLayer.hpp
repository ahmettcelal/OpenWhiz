/*
 * owDistanceLayer.hpp
 *
 *  Created on: Apr 11, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owLayer.hpp"
#include <cmath>
#include <sstream>
#include <string>

namespace ow {

/**
 * @class owDistanceLayer
 * @brief Computes Euclidean distances between input vectors and learnable anchor points.
 * 
 * The owDistanceLayer measures the similarity (in terms of distance) between the 
 * input data and a set of internal reference vectors called "anchors". This is 
 * frequently used in Radial Basis Function (RBF) networks and clustering tasks.
 * 
 * @section math_sec Mathematical Definition
 * - **Forward Pass**: \f$ D_{b,c} = \sqrt{\sum_{i=1}^{I} (X_{b,i} - A_{c,i})^2 + \epsilon} \f$
 *   where \f$ X \f$ is the input, \f$ A \f$ are the learnable anchors, and \f$ \epsilon \f$ 
 *   is a small constant for numerical stability.
 * - **Backward Pass**:
 *   - Gradient with respect to Input: \f$ \frac{\partial D_{b,c}}{\partial X_{b,i}} = \frac{X_{b,i} - A_{c,i}}{D_{b,c}} \f$
 *   - Gradient with respect to Anchor: \f$ \frac{\partial D_{b,c}}{\partial A_{c,i}} = \frac{A_{c,i} - X_{b,i}}{D_{b,c}} \f$
 *   - Total gradients are computed by accumulating these local derivatives weighted by 
 *     the incoming `outputGradient` (\f$ \frac{\partial L}{\partial D} \f$).
 */
class owDistanceLayer : public owLayer {
public:
    /**
     * @brief Constructor for owDistanceLayer.
     * @param inputSize Number of input features (\f$ I \f$).
     * @param numAnchors Number of reference anchor points (\f$ C \f$).
     */
    owDistanceLayer(size_t inputSize, size_t numAnchors) 
        : m_inputSize(inputSize), m_numAnchors(numAnchors) {
        m_layerName = "Distance Layer";
        m_anchors = owTensor<float, 2>::Random({numAnchors, inputSize}, -1.0f, 1.0f);
        m_anchorGradients = owTensor<float, 2>(numAnchors, inputSize);
    }

    /**
     * @brief Returns the expected input feature size.
     */
    size_t getInputSize() const override { return m_inputSize; }

    /**
     * @brief Returns the number of anchor points (output size).
     */
    size_t getOutputSize() const override { return m_numAnchors; }

    /**
     * @brief Resizes the number of anchors and reinitializes their positions.
     * @param num New number of anchors.
     */
    void setNeuronNum(size_t num) override {
        m_numAnchors = num;
        m_anchors = owTensor<float, 2>::Random({m_numAnchors, m_inputSize}, -1.0f, 1.0f);
        m_anchorGradients = owTensor<float, 2>(m_numAnchors, m_inputSize);
    }

    /**
     * @brief Performs forward pass: computes Euclidean distance to each anchor.
     * @param input Input tensor of shape [Batch, InputSize].
     * @return Output tensor of shape [Batch, NumAnchors].
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        m_lastInput = input;
        size_t batch = input.shape()[0];
        owTensor<float, 2> output(batch, m_numAnchors);

        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < m_numAnchors; ++c) {
                float distSq = 0;
                for (size_t i = 0; i < m_inputSize; ++i) {
                    float diff = input(b, i) - m_anchors(c, i);
                    distSq += diff * diff;
                }
                output(b, c) = std::sqrt(distSq + 1e-7f);
            }
        }
        m_lastOutput = output;
        return output;
    }

    /**
     * @brief Performs backward pass: computes gradients for inputs and anchor positions.
     * @param outputGradient Gradient from the following layer.
     * @return Gradient with respect to the input tensor.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        size_t batch = m_lastInput.shape()[0];
        owTensor<float, 2> inputGradient(batch, m_inputSize);
        inputGradient.setZero();
        m_anchorGradients.setZero();

        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < m_numAnchors; ++c) {
                float dist = m_lastOutput(b, c);
                float gradOut = outputGradient(b, c);
                
                for (size_t i = 0; i < m_inputSize; ++i) {
                    float diff = m_lastInput(b, i) - m_anchors(c, i);
                    float dDist_dAnchor = -diff / dist;
                    float dDist_dInput = diff / dist;
                    
                    m_anchorGradients(c, i) += gradOut * dDist_dAnchor;
                    inputGradient(b, i) += gradOut * dDist_dInput;
                }
            }
        }
        return inputGradient;
    }

    /**
     * @brief Updates anchor positions using the configured optimizer.
     */
    void train() override {
        if (m_optimizer) {
            applyRegularization(m_anchors, m_anchorGradients);
            m_optimizer->update(m_anchors, m_anchorGradients);
        }
    }

    float* getParamsPtr() override { return m_anchors.data(); }
    float* getGradsPtr() override { return m_anchorGradients.data(); }
    size_t getParamsCount() override { return m_anchors.size(); }

    /**
     * @brief Creates a deep copy of the distance layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owDistanceLayer>(m_inputSize, m_numAnchors);
        copy->m_anchors = m_anchors;
        copy->m_layerName = m_layerName;
        return copy;
    }

    /**
     * @brief Serializes the layer configuration and anchors to XML.
     */
    std::string toXML() const override {
        std::stringstream ss;
        ss << "<InputSize>" << m_inputSize << "</InputSize>\n";
        ss << "<NumAnchors>" << m_numAnchors << "</NumAnchors>\n";
        ss << "<Anchors>" << m_anchors.toString() << "</Anchors>\n";
        return ss.str();
    }

    /**
     * @brief Deserializes the layer configuration and anchors from XML.
     */
    void fromXML(const std::string& xml) override {
        m_inputSize = std::stoul(getTagContent(xml, "InputSize"));
        m_numAnchors = std::stoul(getTagContent(xml, "NumAnchors"));
        m_anchors = owTensor<float, 2>(m_numAnchors, m_inputSize);
        m_anchorGradients = owTensor<float, 2>(m_numAnchors, m_inputSize);
        m_anchors.fromString(getTagContent(xml, "Anchors"));
    }

private:
    size_t m_inputSize, m_numAnchors;
    owTensor<float, 2> m_anchors, m_anchorGradients, m_lastInput, m_lastOutput;
};

} // namespace ow
