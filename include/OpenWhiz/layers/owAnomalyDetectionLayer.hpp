/*
 * owAnomalyDetectionLayer.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <cmath>
#include <string>

namespace ow {

/**
 * @class owAnomalyDetectionLayer
 * @brief Detects and suppresses anomalies in the input stream based on Z-score.
 * 
 * The owAnomalyDetectionLayer maintains a running mean and standard deviation of
 * the data it sees. If an input value exceeds a certain number of standard deviations
 * (threshold) from the mean, it is considered an anomaly and replaced with the mean.
 * 
 * Implementation Details:
 * - Uses an online algorithm to update mean and variance (m2) for efficiency.
 * - Outliers are suppressed in the forward pass to prevent them from destabilizing downstream layers.
 * - Z-score calculation: z = (x - mean) / std.
 * 
 * Platform Notes:
 * - Industrial: Essential for cleaning noisy sensor data and handling transient spikes.
 * - Computer/Mobile: Lightweight way to ensure model robustness against unexpected inputs.
 */
class owAnomalyDetectionLayer : public owLayer {
public:
    /**
     * @brief Constructor for owAnomalyDetectionLayer.
     * @param threshold Number of standard deviations to trigger suppression (default 3.0).
     */
    owAnomalyDetectionLayer(float threshold = 3.0f) : m_threshold(threshold), m_mean(0), m_std(1), m_count(0) {
        m_layerName = "Anomaly Detection Layer";
    }

    /**
     * @brief Returns 0 as input size is dynamic.
     */
    size_t getInputSize() const override { return 0; }

    /**
     * @brief Returns 0 as output size is dynamic.
     */
    size_t getOutputSize() const override { return 0; }

    /**
     * @brief Dynamic layers ignore fixed neuron counts.
     */
    void setNeuronNum(size_t num) override { (void)num; }

    /**
     * @brief Creates a deep copy of the layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owAnomalyDetectionLayer>(m_threshold);
        copy->m_mean = m_mean; copy->m_std = m_std; copy->m_count = m_count;
        copy->m_layerName = m_layerName;
        return copy;
    }

    /**
     * @brief Performs forward pass with online statistics update and anomaly suppression.
     * @param input Input tensor.
     * @return Cleaned tensor where anomalies are replaced by the running mean.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        // Online update of mean and std
        for (size_t i = 0; i < input.size(); ++i) {
            float x = input.data()[i];
            m_count++;
            float delta = x - m_mean;
            m_mean += delta / m_count;
            float delta2 = x - m_mean;
            m_m2 += delta * delta2;
        }
        m_std = std::sqrt(m_m2 / (m_count > 1 ? m_count - 1 : 1));

        owTensor<float, 2> output = input;
        for (size_t i = 0; i < output.size(); ++i) {
            float z = (output.data()[i] - m_mean) / (m_std + 1e-7f);
            if (std::abs(z) > m_threshold) output.data()[i] = m_mean; // Suppress anomaly
        }
        return output;
    }

    /**
     * @brief Backward pass. Gradient is passed through unchanged.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override { return outputGradient; }
    
    /**
     * @brief Serializes threshold to XML.
     */
    std::string toXML() const override {
        return "<Threshold>" + std::to_string(m_threshold) + "</Threshold>";
    }

    /**
     * @brief Deserializes threshold from XML.
     */
    void fromXML(const std::string& xml) override {
        m_threshold = std::stof(getTagContent(xml, "Threshold"));
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
    float m_threshold, m_mean, m_std, m_m2 = 0; size_t m_count;
};

} // namespace ow
