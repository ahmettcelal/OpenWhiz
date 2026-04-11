/*
 * owPrincipalComponentAnalysisLayer.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <cmath>
#include <sstream>

namespace ow {

/**
 * @class owPrincipalComponentAnalysisLayer
 * @brief Reduces dimensionality of input features using Principal Component Analysis (PCA).
 * 
 * This layer projects high-dimensional data into a lower-dimensional subspace while preserving 
 * as much variance as possible. It is a non-trainable layer that requires fitting on a dataset.
 * 
 * @details
 * **Implementation Details:**
 * - Implements the Power Iteration method with Deflation to compute the top `k` eigenvectors.
 * - Centers the data by subtracting the mean before projection.
 * - Computes components based on the covariance matrix of the provided fitting data.
 * 
 * **Unique Features:**
 * - Manual implementation of PCA suitable for systems without external linear algebra libraries.
 * - Orthogonal projection ensures that components are uncorrelated.
 * 
 * **Platform-Specific Notes:**
 * - **Computer:** Fitting can be slow for very large datasets/dimensions; inference is a simple dot product.
 * - **Mobile/Web:** Ideal for reducing feature size before sending data to a more complex model, 
 *   saving computation and memory.
 * - **Industrial:** Useful for anomaly detection by monitoring the reconstruction error or 
 *   simplifying sensor data.
 */
class owPrincipalComponentAnalysisLayer : public owLayer {
public:
    /**
     * @brief Constructs a PCA layer.
     * @param inputSize The number of features in the input.
     * @param numComponents The number of principal components to keep (output size).
     */
    owPrincipalComponentAnalysisLayer(size_t inputSize, size_t numComponents) 
        : m_inputSize(inputSize), m_numComponents(numComponents), 
          m_mean(1, inputSize), m_components(inputSize, numComponents) {
        m_layerName = "Principal Component Analysis Layer";
        m_mean.setZero();
        m_components.setZero(); // Should be fitted
    }

    size_t getInputSize() const override { return m_inputSize; }
    size_t getOutputSize() const override { return m_numComponents; }
    void setNeuronNum(size_t num) override {
        m_numComponents = num;
        m_components = owTensor<float, 2>(m_inputSize, m_numComponents);
        m_components.setZero();
    }

    /**
     * @brief Fits the PCA components using Power Iteration.
     */
    void fit(const owTensor<float, 2>& data) {
        size_t n = data.shape()[0];
        size_t d = data.shape()[1];
        
        // 1. Compute Mean
        m_mean.setZero();
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d; ++j) m_mean(0, j) += data(i, j);
        }
        for (size_t j = 0; j < d; ++j) m_mean(0, j) /= n;

        // 2. Center Data
        owTensor<float, 2> centered(data.shape());
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d; ++j) centered(i, j) = data(i, j) - m_mean(0, j);
        }

        // 3. Compute Covariance Matrix C = (X^T * X) / (n-1)
        auto cov = centered.transpose().dot(centered);
        for (size_t i = 0; i < cov.size(); ++i) cov.data()[i] /= (n - 1);

        // 4. Power Iteration for top k components
        owTensor<float, 2> currentCov = cov;
        for (size_t k = 0; k < m_numComponents; ++k) {
            owTensor<float, 1> v(d);
            v.setRandom();
            
            // Iterate to find dominant eigenvector
            for (int iter = 0; iter < 100; ++iter) {
                owTensor<float, 1> next_v(d);
                next_v.setZero();
                for (size_t i = 0; i < d; ++i) {
                    for (size_t j = 0; j < d; ++j) next_v(i) += currentCov(i, j) * v(j);
                }
                
                // Normalize
                float norm = 0;
                for (size_t i = 0; i < d; ++i) norm += next_v(i) * next_v(i);
                norm = std::sqrt(norm + 1e-9f);
                for (size_t i = 0; i < d; ++i) v(i) = next_v(i) / norm;
            }

            // Store component
            for (size_t i = 0; i < d; ++i) m_components(i, k) = v(i);

            // Deflation: C_next = C - lambda * v * v^T
            // lambda = v^T * C * v
            float lambda = 0;
            owTensor<float, 1> Cv(d);
            Cv.setZero();
            for (size_t i = 0; i < d; ++i) {
                for (size_t j = 0; j < d; ++j) Cv(i) += currentCov(i, j) * v(j);
                lambda += v(i) * Cv(i);
            }

            for (size_t i = 0; i < d; ++i) {
                for (size_t j = 0; j < d; ++j) {
                    currentCov(i, j) -= lambda * v(i) * v(j);
                }
            }
        }
    }

    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        size_t batch = input.shape()[0];
        owTensor<float, 2> centered(batch, m_inputSize);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t i = 0; i < m_inputSize; ++i) centered(b, i) = input(b, i) - m_mean(0, i);
        }
        return centered.dot(m_components);
    }

    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        // Linear projection backward: grad_in = grad_out * W^T
        return outputGradient.dot(m_components.transpose());
    }

    void train() override {} // PCA is usually not trained via gradient descent here

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
        auto copy = std::make_shared<owPrincipalComponentAnalysisLayer>(m_inputSize, m_numComponents);
        copy->m_mean = m_mean;
        copy->m_components = m_components;
        copy->m_layerName = m_layerName;
        return copy;
    }

    std::string toXML() const override {
        std::stringstream ss;
        ss << "<InputSize>" << m_inputSize << "</InputSize>\n";
        ss << "<NumComponents>" << m_numComponents << "</NumComponents>\n";
        ss << "<Mean>" << m_mean.toString() << "</Mean>\n";
        ss << "<Components>" << m_components.toString() << "</Components>\n";
        return ss.str();
    }

    void fromXML(const std::string& xml) override {
        m_inputSize = std::stoul(getTagContent(xml, "InputSize"));
        m_numComponents = std::stoul(getTagContent(xml, "NumComponents"));
        m_mean = owTensor<float, 2>(1, m_inputSize);
        m_components = owTensor<float, 2>(m_inputSize, m_numComponents);
        m_mean.fromString(getTagContent(xml, "Mean"));
        m_components.fromString(getTagContent(xml, "Components"));
    }

private:
    size_t m_inputSize, m_numComponents;
    owTensor<float, 2> m_mean, m_components;
};

} // namespace ow
