/*
 * owMultiHeadAttentionLayer.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>
#include <sstream>

namespace ow {

/**
 * @class owMultiHeadAttentionLayer
 * @brief Implements the Multi-Head Attention mechanism as used in Transformer architectures.
 * 
 * Multi-Head Attention allows the model to jointly attend to information from different representation 
 * subspaces at different positions. This is achieved by running multiple "heads" of scaled dot-product 
 * attention in parallel.
 * 
 * @details
 * **Implementation Details:**
 * - Projects the input into Query (Q), Key (K), and Value (V) spaces using learned weight matrices.
 * - Computes scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V.
 * - Concatenates the results from all heads and applies a final linear projection.
 * - Current implementation uses a manual loop for heads and attention scores (optimization target).
 * 
 * **Unique Features:**
 * - Scaled dot-product prevents gradients from becoming too small during softmax.
 * - Parallel attention heads capture diverse patterns within the same sequence.
 * 
 * **Platform-Specific Notes:**
 * - **Computer:** Highly parallelizable; performance can be significantly improved with BLAS libraries.
 * - **Mobile/Web:** Computationally intensive for long sequences. Recommended to use with smaller dModel and numHeads.
 * - **Industrial:** Useful for complex pattern recognition in multi-sensor time-series data where cross-sensor dependencies matter.
 */
class owMultiHeadAttentionLayer : public owLayer {
public:
    /**
     * @brief Constructs a Multi-Head Attention layer.
     * @param dModel The dimensionality of the input and output.
     * @param numHeads The number of parallel attention heads. dModel must be divisible by numHeads.
     * @throws std::runtime_error if dModel is not divisible by numHeads.
     */
    owMultiHeadAttentionLayer(size_t dModel, size_t numHeads) 
        : m_dModel(dModel), m_numHeads(numHeads), m_dk(dModel / numHeads) {
        m_layerName = "Multi-Head Attention Layer";
        if (dModel % numHeads != 0) throw std::runtime_error("dModel must be divisible by numHeads");

        // Initialize trainable weights
        m_wQ = owTensor<float, 2>::Random({dModel, dModel}, -0.1f, 0.1f);
        m_wK = owTensor<float, 2>::Random({dModel, dModel}, -0.1f, 0.1f);
        m_wV = owTensor<float, 2>::Random({dModel, dModel}, -0.1f, 0.1f);
        m_wO = owTensor<float, 2>::Random({dModel, dModel}, -0.1f, 0.1f);

        m_gQ = owTensor<float, 2>(dModel, dModel);
        m_gK = owTensor<float, 2>(dModel, dModel);
        m_gV = owTensor<float, 2>(dModel, dModel);
        m_gO = owTensor<float, 2>(dModel, dModel);
    }

    size_t getInputSize() const override { return m_dModel; }
    size_t getOutputSize() const override { return m_dModel; }
    void setNeuronNum(size_t num) override { (void)num; }
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owMultiHeadAttentionLayer>(m_dModel, m_numHeads);
        copy->m_wQ = m_wQ; copy->m_wK = m_wK; copy->m_wV = m_wV; copy->m_wO = m_wO;
        copy->m_layerName = m_layerName;
        return copy;
    }

    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        m_lastInput = input;
        size_t batch = input.shape()[0];

        // 1. Linear Projections (Q, K, V)
        m_Q = input.dot(m_wQ);
        m_K = input.dot(m_wK);
        m_V = input.dot(m_wV);

        // 2. Scaled Dot-Product Attention for each head
        float scale = 1.0f / std::sqrt(static_cast<float>(m_dk));
        owTensor<float, 2> combinedHeads(batch, m_dModel);

        for (size_t h = 0; h < m_numHeads; ++h) {
            size_t offset = h * m_dk;
            for (size_t i = 0; i < batch; ++i) {
                std::vector<float> scores(batch);
                float maxScore = -1e9f;
                for (size_t j = 0; j < batch; ++j) {
                    float dot = 0;
                    for (size_t k = 0; k < m_dk; ++k) dot += m_Q(i, offset + k) * m_K(j, offset + k);
                    scores[j] = dot * scale;
                    maxScore = std::max(maxScore, scores[j]);
                }
                float sumExp = 0;
                for (size_t j = 0; j < batch; ++j) { scores[j] = std::exp(scores[j] - maxScore); sumExp += scores[j]; }
                for (size_t j = 0; j < batch; ++j) scores[j] /= sumExp;

                for (size_t k = 0; k < m_dk; ++k) {
                    float val = 0;
                    for (size_t j = 0; j < batch; ++j) val += scores[j] * m_V(j, offset + k);
                    combinedHeads(i, offset + k) = val;
                }
            }
        }

        m_lastCombined = combinedHeads;
        return combinedHeads.dot(m_wO);
    }

    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        m_gO = m_lastCombined.transpose().dot(outputGradient);
        owTensor<float, 2> dCombined = outputGradient.dot(m_wO.transpose());
        m_gQ = m_lastInput.transpose().dot(dCombined);
        m_gK = m_lastInput.transpose().dot(dCombined);
        m_gV = m_lastInput.transpose().dot(dCombined);
        return dCombined.dot(m_wQ.transpose()); 
    }

    void train() override {
        if (m_optimizer) {
            m_optimizer->update(m_wQ, m_gQ);
            m_optimizer->update(m_wK, m_gK);
            m_optimizer->update(m_wV, m_gV);
            m_optimizer->update(m_wO, m_gO);
        }
    }

    std::string toXML() const override {
        std::stringstream ss;
        ss << "<DModel>" << m_dModel << "</DModel><Heads>" << m_numHeads << "</Heads>";
        return ss.str();
    }
    void fromXML(const std::string& xml) override {
        m_dModel = std::stoul(getTagContent(xml, "DModel"));
        m_numHeads = std::stoul(getTagContent(xml, "Heads"));
        m_dk = m_dModel / m_numHeads;
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
	size_t m_dModel, m_numHeads, m_dk;
	owTensor<float, 2> m_wQ, m_wK, m_wV, m_wO;
	owTensor<float, 2> m_gQ, m_gK, m_gV, m_gO;
	owTensor<float, 2> m_Q, m_K, m_V, m_lastInput, m_lastCombined;
};

} // namespace ow
