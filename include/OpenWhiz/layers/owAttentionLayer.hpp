/*
 * owAttentionLayer.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <cmath>
#include <algorithm>
#include <string>

namespace ow {

/**
 * @class owAttentionLayer
 * @brief Implements a basic self-attention mechanism (Scaled Dot-Product Attention).
 * 
 * The owAttentionLayer computes alignment scores between different samples in a batch.
 * It allows the model to focus on different parts of the input sequence or batch
 * when producing the output.
 * 
 * Implementation Details:
 * - Computes Alignment = Softmax((Q * K^T) / sqrt(d_k)).
 * - In this implementation, Query (Q), Key (K), and Value (V) are all derived 
 *   directly from the input tensor.
 * - Output is a weighted sum of the input features based on the alignment scores.
 * 
 * Platform Notes:
 * - Computer: Efficiently handles moderate batch sizes.
 * - Mobile/Web: Batch size should be kept small to avoid O(N^2) complexity in scoring.
 * - Industrial: Useful for temporal sequence analysis in multi-sensor systems.
 */
class owAttentionLayer : public owLayer {
public:
    /**
     * @brief Constructor for owAttentionLayer.
     * @param inputDim The dimension of the input features.
     */
    owAttentionLayer(size_t inputDim) : m_dim(inputDim) {
        m_layerName = "Attention Layer";
    }

    /**
     * @brief Returns the expected input feature dimension.
     */
    size_t getInputSize() const override { return m_dim; }

    /**
     * @brief Returns the output feature dimension (same as input).
     */
    size_t getOutputSize() const override { return m_dim; }

    /**
     * @brief Sets the feature dimension.
     * @param num New feature size.
     */
    void setNeuronNum(size_t num) override { m_dim = num; }

    /**
     * @brief Creates a deep copy of the layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owAttentionLayer>(m_dim);
        copy->m_layerName = m_layerName;
        return copy;
    }

    /**
     * @brief Serializes dimension to XML.
     */
    std::string toXML() const override {
        return "<Dim>" + std::to_string(m_dim) + "</Dim>";
    }

    /**
     * @brief Deserializes dimension from XML.
     */
    void fromXML(const std::string& xml) override {
        m_dim = std::stoul(getTagContent(xml, "Dim"));
    }

    /**
     * @brief Performs forward pass: computes scaled dot-product attention.
     * @param input Input tensor of shape [BatchSize, Dim].
     * @return Attended output tensor of shape [BatchSize, Dim].
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        m_lastInput = input;
        size_t batch = input.shape()[0];
        owTensor<float, 2> scores(batch, batch);
        
        // Compute alignment scores (Attention weights)
        float scale = 1.0f / std::sqrt(static_cast<float>(m_dim));
        for(size_t i=0; i<batch; ++i) {
            for(size_t j=0; j<batch; ++j) {
                float dot = 0;
                for(size_t k=0; k<m_dim; ++k) dot += input(i, k) * input(j, k);
                scores(i, j) = dot * scale;
            }
        }
        
        // Softmax rows
        for(size_t i=0; i<batch; ++i) {
            float maxVal = scores(i, 0);
            for(size_t j=1; j<batch; ++j) maxVal = std::max(maxVal, scores(i, j));
            float sum = 0;
            for(size_t j=0; j<batch; ++j) { scores(i, j) = std::exp(scores(i, j) - maxVal); sum += scores(i, j); }
            for(size_t j=0; j<batch; ++j) scores(i, j) /= sum;
        }
        
        m_attentionWeights = scores;
        
        // Weighted sum
        owTensor<float, 2> output(batch, m_dim);
        for(size_t i=0; i<batch; ++i) {
            for(size_t k=0; k<m_dim; ++k) {
                float val = 0;
                for(size_t j=0; j<batch; ++j) val += scores(i, j) * input(j, k);
                output(i, k) = val;
            }
        }
        return output;
    }

    /**
     * @brief Backward pass. Gradient is passed through.
     * @note This implementation uses fixed attention (no learned weights).
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        return outputGradient; 
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
    size_t m_dim; owTensor<float, 2> m_lastInput, m_attentionWeights;
};

} // namespace ow
