/*
 * owProjectionLayer.hpp
 *
 *  Created on: Apr 11, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owLayer.hpp"
#include "../optimizers/owOptimizer.hpp"
#include <random>
#include <chrono>

namespace ow {

/**
 * @class owProjectionLayer
 * @brief Performs a learnable linear projection of input data: \f$ Y = XW \f$.
 * 
 * The owProjectionLayer maps input vectors from a high-dimensional space to a 
 * lower (or higher) dimensional latent space using a weight matrix \f$ W \f$.
 * 
 * @section math_sec Mathematical Definition
 * - **Forward Pass**: \f$ Y = X \cdot W \f$, where \f$ X \in \mathbb{R}^{B \times I} \f$, 
 *   \f$ W \in \mathbb{R}^{I \times O} \f$, and \f$ Y \in \mathbb{R}^{B \times O} \f$.
 *   (\f$ B \f$: Batch size, \f$ I \f$: Input size, \f$ O \f$: Output size).
 * - **Backward Pass (Gradients)**:
 *   - Weight Gradient: \f$ \frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y} \f$
 *   - Input Gradient: \f$ \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T \f$
 * 
 * @section comparison_sec Comparison with other layers
 * - **vs owLinearLayer**: Unlike owLinearLayer, this layer **does not include a bias term** (\f$ b \f$) 
 *   and does not have an integrated activation function. It is a "pure" linear transformation, 
 *   making it ideal for dimensionality reduction where centering is handled elsewhere (e.g., Normalization).
 * - **vs owPrincipalComponentAnalysisLayer (PCA)**: While PCA finds a projection based on 
 *   **unsupervised variance maximization** (deterministic), owProjectionLayer's weights are 
 *   **learnable parameters** updated via backpropagation. This allows the projection to be 
 *   optimized specifically for the target task (supervised or self-supervised clustering).
 */
class owProjectionLayer : public owLayer {
public:
    /**
     * @brief Constructor for owProjectionLayer.
     * @param inputSize Number of input features (\f$ I \f$).
     * @param outputSize Number of projected dimensions (\f$ O \f$).
     */
    owProjectionLayer(size_t inputSize, size_t outputSize) 
        : m_inputSize(inputSize), m_outputSize(outputSize),
          m_weights(inputSize, outputSize),
          m_weightGradients(inputSize, outputSize) {
        m_layerName = "Projection Layer";
        initializeWeights();
    }

    /**
     * @brief Returns the expected input feature size.
     */
    size_t getInputSize() const override { return m_inputSize; }

    /**
     * @brief Returns the projected dimensionality.
     */
    size_t getOutputSize() const override { return m_outputSize; }

    /**
     * @brief Resizes the output dimensionality and reinitializes weights.
     * @param num New number of output dimensions.
     */
    void setNeuronNum(size_t num) override {
        m_outputSize = num;
        m_weights = owTensor<float, 2>(m_inputSize, m_outputSize);
        m_weightGradients = owTensor<float, 2>(m_inputSize, m_outputSize);
        initializeWeights();
    }

    /**
     * @brief Initializes weights using Xavier (Glorot) initialization.
     * 
     * Uses \f$ \text{Var}(W) = \frac{2}{n_{in} + n_{out}} \f$ to maintain signal variance.
     */
    void initializeWeights() {
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(static_cast<unsigned int>(seed));
        float range = std::sqrt(6.0f / (m_inputSize + m_outputSize));
        std::uniform_real_distribution<float> distribution(-range, range);
        for (size_t i = 0; i < m_weights.size(); ++i) {
            m_weights.data()[i] = distribution(generator);
        }
    }

    /**
     * @brief Performs forward pass: \f$ Y = XW \f$.
     * @param input Input tensor of shape [Batch, InputSize].
     * @return Projected tensor of shape [Batch, OutputSize].
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        m_lastInput = input;
        return input.dot(m_weights);
    }

    /**
     * @brief Performs backward pass: computes gradients for weights and input features.
     * @param outputGradient Gradient from the following layer.
     * @return Gradient with respect to the input tensor.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        // dW = input^T * dO
        auto inputT = m_lastInput.transpose();
        auto dW = inputT.dot(outputGradient);
        m_weightGradients = dW;

        // dX = dO * W^T
        auto weightsT = m_weights.transpose();
        return outputGradient.dot(weightsT);
    }

    /**
     * @brief Updates projection weights using the configured optimizer.
     */
    void train() override {
        if (m_optimizer) {
            applyRegularization(m_weights, m_weightGradients);
            m_optimizer->update(m_weights, m_weightGradients);
        }
    }

    float* getParamsPtr() override { return m_weights.data(); }
    float* getGradsPtr() override { return m_weightGradients.data(); }
    size_t getParamsCount() override { return m_weights.size(); }

    /**
     * @brief Creates a deep copy of the projection layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owProjectionLayer>(m_inputSize, m_outputSize);
        copy->m_weights = m_weights;
        copy->m_layerName = m_layerName;
        return copy;
    }

    /**
     * @brief Serializes the layer configuration and weights to XML.
     */
    std::string toXML() const override {
        std::stringstream ss;
        ss << "<InputSize>" << m_inputSize << "</InputSize>\n";
        ss << "<OutputSize>" << m_outputSize << "</OutputSize>\n";
        ss << "<Weights>" << m_weights.toString() << "</Weights>\n";
        return ss.str();
    }

    /**
     * @brief Deserializes the layer configuration and weights from XML.
     */
    void fromXML(const std::string& xml) override {
        m_inputSize = std::stoul(getTagContent(xml, "InputSize"));
        m_outputSize = std::stoul(getTagContent(xml, "OutputSize"));
        m_weights = owTensor<float, 2>(m_inputSize, m_outputSize);
        m_weightGradients = owTensor<float, 2>(m_inputSize, m_outputSize);
        m_weights.fromString(getTagContent(xml, "Weights"));
    }

private:
    size_t m_inputSize, m_outputSize;
    owTensor<float, 2> m_weights, m_weightGradients, m_lastInput;
};

} // namespace ow
