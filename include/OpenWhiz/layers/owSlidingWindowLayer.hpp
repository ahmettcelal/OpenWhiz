/*
 * owSlidingWindowLayer.hpp
 *
 *  Created on: Apr 11, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owLayer.hpp"
#include <deque>

namespace ow {

/**
 * @class owSlidingWindowLayer
 * @brief A stateful layer that transforms a single column of input into a temporal sliding window.
 * 
 * This layer is specifically designed for Time-Series Forecasting. It maintains an internal 
 * rolling buffer (history) of past values for a specific target column, allowing the network 
 * to learn from temporal dependencies without requiring the user to manually restructure 
 * the dataset into windows.
 * 
 * **Functionality:**
 * 1. Extracts a specific column (targetIdx) from the input batch.
 * 2. Generates a look-back window of size 'windowSize' using internal history.
 * 3. Optionally appends the entire current feature vector to the windowed output.
 * 
 * **Output Structure (per sample):**
 * `[t-1, t-2, ..., t-windowSize, current_features...]`
 * 
 * **Calculation Details:**
 * - **History Management:** Uses a `std::deque` to store the last `windowSize * dilation` points.
 * - **Dilation:** Allows skipping steps in the history (e.g., dilation=2 with windowSize=3 
 *   would look at points t-2, t-4, t-6).
 * - **Statefulness:** The layer remembers values between `forward()` calls. Call `reset()` 
 *   to clear this history (essential between independent sequences or at the start of an epoch).
 */
class owSlidingWindowLayer : public owLayer {
public:
    /**
     * @brief Constructor for owSlidingWindowLayer.
     * @param windowSize The number of historical points to include in the output vector.
     * @param dilation The gap between historical points (1 = consecutive, 2 = every other point).
     * @param targetIdx The index of the column to be windowed. If -1, the last column is used.
     * @param includeCurrent If true, the full input feature vector at time 't' is appended to the output.
     */
    owSlidingWindowLayer(size_t windowSize = 5, size_t dilation = 1, int targetIdx = -1, bool includeCurrent = true) 
        : m_windowSize(windowSize), m_dilation(dilation), m_targetIdx(targetIdx), m_inputFeatures(1), m_includeCurrent(includeCurrent) {
        m_layerName = "Sliding Window Layer";
    }

    /** @return The number of expected input features (dynamic, updated during forward). */
    size_t getInputSize() const override { return m_inputFeatures; } 

    /** 
     * @brief Calculates the output vector size.
     * Formula: windowSize + (includeCurrent ? inputFeatures : 0)
     */
    size_t getOutputSize() const override { return m_windowSize + (m_includeCurrent ? m_inputFeatures : 0); }
    
    /** @brief Updates the internal input feature count. */
    void setNeuronNum(size_t num) override { m_inputFeatures = num; }

    /** @brief Clears the internal history buffer. Essential for starting new, unrelated sequences. */
    void reset() override {
        m_history.clear();
    }

    /** 
     * @brief Creates a deep copy of the layer configuration.
     * @note Internal history is NOT cloned to ensure the new instance starts with a fresh state.
     */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owSlidingWindowLayer>(m_windowSize, m_dilation, m_targetIdx, m_includeCurrent);
        copy->m_layerName = m_layerName;
        copy->m_inputFeatures = m_inputFeatures;
        return copy;
    }

    /** 
     * @brief Performs the forward pass by constructing temporal windows.
     * 
     * **Logic:**
     * 1. Validates the target column index (`targetIdx`).
     * 2. Iterates through each sample in the `batchSize`:
     *    a. For each window step `w` [0 to windowSize-1]:
     *       - Calculates the look-back index: `history.size() - ((w + 1) * dilation)`.
     *       - If the index exists in history, maps it to `output(sample, w)`.
     *       - Otherwise, pads with `0.0f` (cold-start handling).
     *    b. Updates the rolling history buffer with the current `currentTargetVal`.
     *    c. If `includeCurrent` is enabled, copies all features of the current sample 
     *       to the remaining slots in the output vector.
     * 
     * @param input Tensor of shape [BatchSize, Features].
     * @return Tensor of shape [BatchSize, OutputSize].
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        size_t batchSize = input.shape()[0];
        size_t totalFeatures = input.shape()[1];
        m_inputFeatures = totalFeatures; 

        int idx = m_targetIdx;
        if (idx == -1) idx = (int)totalFeatures - 1;
        if (idx < 0 || (size_t)idx >= totalFeatures) idx = 0;

        owTensor<float, 2> output(batchSize, getOutputSize());
        
        for (size_t i = 0; i < batchSize; ++i) {
            float currentTargetVal = input(i, (size_t)idx);

            // 1. Map history to the first part of the output vector
            // Output indices [0 ... m_windowSize-1]
            for (size_t w = 0; w < m_windowSize; ++w) {
                size_t lookbackSteps = (w + 1) * m_dilation;
                if (lookbackSteps <= m_history.size()) {
                    // History is a deque where the back is the most recent (t-1)
                    output(i, w) = m_history[m_history.size() - lookbackSteps];
                } else {
                    output(i, w) = 0.0f; // Zero padding for steps beyond available history
                }
            }
            
            // 2. Update the rolling buffer with the current value (t)
            m_history.push_back(currentTargetVal);
            if (m_history.size() > m_windowSize * m_dilation) {
                m_history.pop_front(); // Maintain maximum required buffer size
            }

            // 3. Append current feature vector to the second part of the output vector
            // Output indices [m_windowSize ... m_windowSize + totalFeatures - 1]
            if (m_includeCurrent) {
                for (size_t f = 0; f < totalFeatures; ++f) {
                    output(i, m_windowSize + f) = input(i, f);
                }
            }
        }
        return output;
    }

    /** 
     * @brief Propagates gradients back to the input features.
     * 
     * **Logic:**
     * Since the sliding window (look-back part) is an indexing operation 
     * involving past samples, it does not have trainable weights. 
     * 
     * If `includeCurrent` is true, the gradients for the features at 
     * time 't' are passed back to the input. The historical window 
     * indices do not contribute to the current input's gradient 
     * because they belong to previous time steps.
     * 
     * @param outputGradient Gradient of the loss with respect to this layer's output.
     * @return Gradient of the loss with respect to this layer's input.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        size_t batchSize = outputGradient.shape()[0];
        owTensor<float, 2> inputGradient(batchSize, m_inputFeatures);
        
        if (m_includeCurrent) {
            for (size_t i = 0; i < batchSize; ++i) {
                for (size_t f = 0; f < m_inputFeatures; ++f) {
                    // Direct mapping: Input Features -> Output[m_windowSize + f]
                    inputGradient(i, f) = outputGradient(i, m_windowSize + f);
                }
            }
        }
        return inputGradient;
    }

    /** @return XML formatted configuration string for serialization. */
    std::string toXML() const override {
        std::stringstream ss;
        ss << "<WindowSize>" << m_windowSize << "</WindowSize>\n";
        ss << "<Dilation>" << m_dilation << "</Dilation>\n";
        ss << "<TargetIdx>" << m_targetIdx << "</TargetIdx>\n";
        ss << "<InputFeatures>" << m_inputFeatures << "</InputFeatures>\n";
        ss << "<IncludeCurrent>" << (m_includeCurrent ? 1 : 0) << "</IncludeCurrent>\n";
        return ss.str();
    }

    /** @brief Reconstructs the layer configuration from an XML string. */
    void fromXML(const std::string& xml) override {
        m_windowSize = std::stoul(getTagContent(xml, "WindowSize"));
        m_dilation = std::stoul(getTagContent(xml, "Dilation"));
        m_targetIdx = std::stoi(getTagContent(xml, "TargetIdx"));
        m_inputFeatures = std::stoul(getTagContent(xml, "InputFeatures"));
        m_includeCurrent = std::stoi(getTagContent(xml, "IncludeCurrent")) != 0;
        reset(); // Clear history when loading new configuration
    }

    /** @brief This layer has no trainable parameters. */
    void train() override {}
    float* getParamsPtr() override { return nullptr; }
    float* getGradsPtr() override { return nullptr; }
    size_t getParamsCount() override { return 0; }

    /** @brief Sets whether current features should be included in the output vector. */
    void setIncludeCurrent(bool include) { m_includeCurrent = include; }

private:
    size_t m_windowSize;    ///< Number of past samples in the window.
    size_t m_dilation;      ///< Spacing between samples in the history.
    int m_targetIdx;        ///< Target column index to window.
    size_t m_inputFeatures; ///< Number of features in the input stream.
    bool m_includeCurrent;  ///< Flag to include current time step features.
    std::deque<float> m_history; ///< Stateful rolling buffer for past values.
};

} // namespace ow
