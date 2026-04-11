/*
 * owRankingLayer.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <sstream>

namespace ow {

/**
 * @class owRankingLayer
 * @brief Implements a pairwise ranking mechanism with shared weights.
 * 
 * This layer is designed for Learning to Rank (LTR) tasks. It takes a concatenated input 
 * of two items (x1, x2) and computes a score for each using the same set of weights.
 * 
 * @details
 * **Implementation Details:**
 * - Expects input of size 2 * itemDim (concatenation of feature vectors for two items).
 * - Uses shared weights for both items to ensure consistent scoring (W * x1 and W * x2).
 * - Output is a 2-element vector containing [score1, score2].
 * - Gradients from both scores are accumulated into the shared weights.
 * 
 * **Unique Features:**
 * - Shared weights ensure that the relative ranking doesn't depend on the order of items in the input.
 * - Optimized for pairwise comparison tasks like recommendation systems or search ranking.
 * 
 * **Platform-Specific Notes:**
 * - **Computer/Mobile/Web:** Efficiently handles pairwise comparisons in a single pass.
 * - **Industrial:** Useful for comparing quality or performance of two different processes or 
 *   product configurations.
 */
class owRankingLayer : public owLayer {
public:
    /**
     * @brief Constructs a Ranking layer.
     * @param itemDim The dimensionality of a single item's feature vector.
     */
    owRankingLayer(size_t itemDim) : m_itemDim(itemDim) {
        m_layerName = "Ranking Layer";
        m_weights = owTensor<float, 2>(itemDim, 1);
        m_weights.setRandom(-0.1f, 0.1f);
        m_weightGradients = owTensor<float, 2>(itemDim, 1);
    }

    size_t getInputSize() const override { return 2 * m_itemDim; }
    size_t getOutputSize() const override { return 2; }
    void setNeuronNum(size_t num) override {
        m_itemDim = num / 2;
        m_weights = owTensor<float, 2>(m_itemDim, 1);
        m_weights.setRandom(-0.1f, 0.1f);
        m_weightGradients = owTensor<float, 2>(m_itemDim, 1);
    }

    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        m_lastInput = input;
        size_t batch = input.shape()[0];
        owTensor<float, 2> output(batch, 2);

        for (size_t b = 0; b < batch; ++b) {
            float s1 = 0, s2 = 0;
            for (size_t i = 0; i < m_itemDim; ++i) {
                s1 += input(b, i) * m_weights(i, 0);
                s2 += input(b, m_itemDim + i) * m_weights(i, 0);
            }
            output(b, 0) = s1;
            output(b, 1) = s2;
        }
        return output;
    }

    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        size_t batch = m_lastInput.shape()[0];
        owTensor<float, 2> inputGradient(batch, 2 * m_itemDim);
        m_weightGradients.setZero();

        for (size_t b = 0; b < batch; ++b) {
            float ds1 = outputGradient(b, 0);
            float ds2 = outputGradient(b, 1);
            
            for (size_t i = 0; i < m_itemDim; ++i) {
                // Gradient w.r.t weights (accumulated from both items in pair)
                m_weightGradients(i, 0) += ds1 * m_lastInput(b, i);
                m_weightGradients(i, 0) += ds2 * m_lastInput(b, m_itemDim + i);
                
                // Gradient w.r.t input
                inputGradient(b, i) = ds1 * m_weights(i, 0);
                inputGradient(b, m_itemDim + i) = ds2 * m_weights(i, 0);
            }
        }
        return inputGradient;
    }

    void train() override {
        if (m_optimizer) {
            applyRegularization(m_weights, m_weightGradients);
            m_optimizer->update(m_weights, m_weightGradients);
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


    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owRankingLayer>(m_itemDim);
        copy->m_weights = m_weights;
        copy->m_layerName = m_layerName;
        return copy;
    }

    std::string toXML() const override {
        std::stringstream ss;
        ss << "<ItemDim>" << m_itemDim << "</ItemDim>\n";
        ss << "<Weights>" << m_weights.toString() << "</Weights>\n";
        ss << "<RegType>" << static_cast<int>(m_regType) << "</RegType>\n";
        ss << "<RegLambda>" << m_regLambda << "</RegLambda>\n";
        return ss.str();
    }

    void fromXML(const std::string& xml) override {
        m_itemDim = std::stoul(getTagContent(xml, "ItemDim"));
        m_weights = owTensor<float, 2>(m_itemDim, 1);
        m_weightGradients = owTensor<float, 2>(m_itemDim, 1);
        m_weights.fromString(getTagContent(xml, "Weights"));
        std::string rt = getTagContent(xml, "RegType");
        if (!rt.empty()) m_regType = static_cast<owRegularizationType>(std::stoi(rt));
        std::string rl = getTagContent(xml, "RegLambda");
        if (!rl.empty()) m_regLambda = std::stof(rl);
    }

private:
    size_t m_itemDim;
    owTensor<float, 2> m_weights, m_weightGradients, m_lastInput;
};

} // namespace ow
