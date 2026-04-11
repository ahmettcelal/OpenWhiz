/*
 * owClusterLayer.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <cmath>
#include <sstream>
#include <string>

namespace ow {

/**
 * @class owClusterLayer
 * @brief Performs centroid-based clustering and outputs distances to cluster centers.
 * 
 * The owClusterLayer maps input vectors to a space defined by a set of learnable
 * centroids. The output of the layer is a vector of Euclidean distances from the
 * input to each centroid. This is similar to the first stage of a Radial Basis
 * Function (RBF) network.
 * 
 * Implementation Details:
 * - Forward: output[b, c] = sqrt(sum((input[b, i] - centroid[c, i])^2)).
 * - Centroids are learnable parameters that move towards the data distribution
 *   during training.
 * - Supports regularization to prevent centroids from collapsing or diverging.
 * 
 * Platform Notes:
 * - Computer: Optimized for high-dimensional feature clustering.
 * - Mobile: Useful for lightweight on-device pattern grouping.
 * - Industrial: Ideal for condition monitoring where "distance from normal state"
 *   centroids can indicate health status.
 */
class owClusterLayer : public owLayer {
public:
    /**
     * @brief Constructor for owClusterLayer.
     * @param inputSize Number of input features.
     * @param numClusters Number of centroids to learn.
     */
    owClusterLayer(size_t inputSize, size_t numClusters) 
        : m_inputSize(inputSize), m_numClusters(numClusters) {
        m_layerName = "Cluster Layer";
        m_centroids = owTensor<float, 2>::Random({numClusters, inputSize}, -1.0f, 1.0f);
        m_centroidGradients = owTensor<float, 2>(numClusters, inputSize);
    }

    /**
     * @brief Returns the expected input feature size.
     */
    size_t getInputSize() const override { return m_inputSize; }

    /**
     * @brief Returns the number of clusters (output size).
     */
    size_t getOutputSize() const override { return m_numClusters; }

    /**
     * @brief Resizes the cluster count and reinitializes centroids.
     * @param num New number of clusters.
     */
    void setNeuronNum(size_t num) override {
        m_numClusters = num;
        m_centroids = owTensor<float, 2>::Random({m_numClusters, m_inputSize}, -1.0f, 1.0f);
        m_centroidGradients = owTensor<float, 2>(m_numClusters, m_inputSize);
    }

    /**
     * @brief Performs forward pass: computes Euclidean distance to each centroid.
     * @param input Input tensor of shape [Batch, InputSize].
     * @return Output tensor of shape [Batch, NumClusters].
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        m_lastInput = input;
        size_t batch = input.shape()[0];
        owTensor<float, 2> output(batch, m_numClusters);

        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < m_numClusters; ++c) {
                float distSq = 0;
                for (size_t i = 0; i < m_inputSize; ++i) {
                    float diff = input(b, i) - m_centroids(c, i);
                    distSq += diff * diff;
                }
                output(b, c) = std::sqrt(distSq + 1e-7f);
            }
        }
        m_lastOutput = output;
        return output;
    }

    /**
     * @brief Performs backward pass, computing gradients for both input and centroids.
     * @param outputGradient Gradient from the next layer.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        size_t batch = m_lastInput.shape()[0];
        owTensor<float, 2> inputGradient(batch, m_inputSize);
        inputGradient.setZero();
        m_centroidGradients.setZero();

        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < m_numClusters; ++c) {
                float dist = m_lastOutput(b, c);
                float gradOut = outputGradient(b, c);
                
                for (size_t i = 0; i < m_inputSize; ++i) {
                    float diff = m_lastInput(b, i) - m_centroids(c, i);
                    float dDist_dCentroid = -diff / dist;
                    float dDist_dInput = diff / dist;
                    
                    m_centroidGradients(c, i) += gradOut * dDist_dCentroid;
                    inputGradient(b, i) += gradOut * dDist_dInput;
                }
            }
        }
        return inputGradient;
    }

    /**
     * @brief Updates centroids using the attached optimizer and applies regularization.
     */
    void train() override {
        if (m_optimizer) {
            applyRegularization(m_centroids, m_centroidGradients);
            m_optimizer->update(m_centroids, m_centroidGradients);
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


    /**
     * @brief Creates a deep copy of the layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owClusterLayer>(m_inputSize, m_numClusters);
        copy->m_centroids = m_centroids;
        copy->m_layerName = m_layerName;
        return copy;
    }

    /**
     * @brief Serializes configuration and centroids to XML.
     */
    std::string toXML() const override {
        std::stringstream ss;
        ss << "<InputSize>" << m_inputSize << "</InputSize>\n";
        ss << "<NumClusters>" << m_numClusters << "</NumClusters>\n";
        ss << "<Centroids>" << m_centroids.toString() << "</Centroids>\n";
        ss << "<RegType>" << static_cast<int>(m_regType) << "</RegType>\n";
        ss << "<RegLambda>" << m_regLambda << "</RegLambda>\n";
        return ss.str();
    }

    /**
     * @brief Deserializes configuration and centroids from XML.
     */
    void fromXML(const std::string& xml) override {
        m_inputSize = std::stoul(getTagContent(xml, "InputSize"));
        m_numClusters = std::stoul(getTagContent(xml, "NumClusters"));
        m_centroids = owTensor<float, 2>(m_numClusters, m_inputSize);
        m_centroidGradients = owTensor<float, 2>(m_numClusters, m_inputSize);
        m_centroids.fromString(getTagContent(xml, "Centroids"));
        std::string rt = getTagContent(xml, "RegType");
        if (!rt.empty()) m_regType = static_cast<owRegularizationType>(std::stoi(rt));
        std::string rl = getTagContent(xml, "RegLambda");
        if (!rl.empty()) m_regLambda = std::stof(rl);
    }

private:
    size_t m_inputSize, m_numClusters;
    owTensor<float, 2> m_centroids, m_centroidGradients, m_lastInput, m_lastOutput;
};

} // namespace ow
