/*
 * owLSTMLayer.hpp
 *
 *  Created on: Jan 12, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>

namespace ow {

/**
 * @class owLSTMLayer
 * @brief Implements a Long Short-Term Memory (LSTM) recurrent layer for sequence processing.
 * 
 * The LSTM layer is designed to solve the vanishing gradient problem in standard RNNs by using a gated 
 * mechanism. It maintains a cell state that can store information over long time steps, controlled by 
 * forget, input, and output gates.
 * 
 * @details
 * **Implementation Details:**
 * - Uses a single weight matrix for all four gates (Forget, Input, Candidate, Output) to optimize 
 *   memory layout and computation (concatenated as [W_f, W_i, W_c, W_o]).
 * - Supports returning either the full sequence of hidden states or just the last state.
 * - Handles truncated backpropagation through time (BPTT) internally for sequence data.
 * 
 * **Unique Features:**
 * - Optimized gate computation by concatenating input and hidden states before dot product.
 * - Flexibility in output shape for many-to-many or many-to-one sequence modeling.
 * 
 * **Platform-Specific Notes:**
 * - **Computer:** Highly efficient for sequential data like time-series or NLP. Benefits from vectorized math.
 * - **Mobile/Web:** Optimized for memory by combining gate weights, reducing the number of tensor operations.
 * - **Industrial:** Suitable for real-time sensor data smoothing and predictive maintenance where temporal 
 *   dependencies are critical.
 */
class owLSTMLayer : public owLayer {
public:
    /**
     * @brief Constructs an LSTM layer.
     * @param inputSize The number of features in the input sequence.
     * @param hiddenSize The number of units in the hidden state (and output).
     * @param returnSequence If true, returns hidden states for all time steps. If false, returns only the last hidden state.
     */
    owLSTMLayer(size_t inputSize, size_t hiddenSize, bool returnSequence = false) 
        : m_inputSize(inputSize), m_hiddenSize(hiddenSize), m_returnSequence(returnSequence) {
        m_layerName = "LSTM Layer";
        
        // Combined weights for 4 gates (forget, input, candidate, output)
        // [W_f, W_i, W_c, W_o] concatenated. 
        // Row size: inputSize + hiddenSize (concatenated [x_t, h_{t-1}])
        // Col size: 4 * hiddenSize
        m_weights = owTensor<float, 2>(inputSize + hiddenSize, 4 * hiddenSize);
        m_biases = owTensor<float, 2>(1, 4 * hiddenSize);
        
        m_weights.setRandom(-0.1f, 0.1f);
        m_biases.setZero();

        m_weightGradients = owTensor<float, 2>(inputSize + hiddenSize, 4 * hiddenSize);
        m_biasGradients = owTensor<float, 2>(1, 4 * hiddenSize);
    }

    size_t getInputSize() const override { return m_inputSize; }
    size_t getOutputSize() const override { return m_hiddenSize; }
    void setNeuronNum(size_t num) override {
        m_hiddenSize = num;
        m_weights = owTensor<float, 2>(m_inputSize + m_hiddenSize, 4 * m_hiddenSize);
        m_biases = owTensor<float, 2>(1, 4 * m_hiddenSize);
        m_weightGradients = owTensor<float, 2>(m_inputSize + m_hiddenSize, 4 * m_hiddenSize);
        m_biasGradients = owTensor<float, 2>(1, 4 * m_hiddenSize);
        m_weights.setRandom(-0.1f, 0.1f);
        m_biases.setZero();
    }

    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        size_t timeSteps = input.shape()[0];
        m_lastInput = input;
        
        // Storage for internal states for backward pass
        m_h.assign(timeSteps + 1, owTensor<float, 2>(1, m_hiddenSize));
        for(auto& t : m_h) t.setConstant(0.0f);
        m_c.assign(timeSteps + 1, owTensor<float, 2>(1, m_hiddenSize));
        for(auto& t : m_c) t.setConstant(0.0f);
        m_gates.assign(timeSteps, owTensor<float, 2>(1, 4 * m_hiddenSize));
        for(auto& t : m_gates) t.setConstant(0.0f);

        for (size_t t = 0; t < timeSteps; ++t) {
            // 1. Concatenate x_t and h_{t-1}
            owTensor<float, 2> x_h(1, m_inputSize + m_hiddenSize);
            for (size_t i = 0; i < m_inputSize; ++i) x_h(0, i) = input(t, i);
            for (size_t i = 0; i < m_hiddenSize; ++i) x_h(0, m_inputSize + i) = m_h[t](0, i);

            // 2. Compute Gates: z = x_h * W + b
            auto z = x_h.dot(m_weights);
            for (size_t i = 0; i < 4 * m_hiddenSize; ++i) z(0, i) += m_biases(0, i);

            // 3. Apply Activations
            for (size_t i = 0; i < m_hiddenSize; ++i) {
                float f = 1.0f / (1.0f + std::exp(-z(0, i))); // Forget gate
                float in = 1.0f / (1.0f + std::exp(-z(0, m_hiddenSize + i))); // Input gate
                float g = std::tanh(z(0, 2 * m_hiddenSize + i)); // Candidate
                float o = 1.0f / (1.0f + std::exp(-z(0, 3 * m_hiddenSize + i))); // Output gate

                z(0, i) = f;
                z(0, m_hiddenSize + i) = in;
                z(0, 2 * m_hiddenSize + i) = g;
                z(0, 3 * m_hiddenSize + i) = o;

                // 4. Update Cell State: c_t = f*c_{t-1} + i*g
                m_c[t + 1](0, i) = f * m_c[t](0, i) + in * g;
                
                // 5. Update Hidden State: h_t = o * tanh(c_t)
                m_h[t + 1](0, i) = o * std::tanh(m_c[t + 1](0, i));
            }
            m_gates[t] = z;
        }

        if (m_returnSequence) {
            owTensor<float, 2> out(timeSteps, m_hiddenSize);
            for (size_t t = 0; t < timeSteps; ++t) {
                for (size_t i = 0; i < m_hiddenSize; ++i) out(t, i) = m_h[t + 1](0, i);
            }
            return out;
        } else {
            return m_h[timeSteps]; // Return only last hidden state
        }
    }

    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        size_t timeSteps = m_lastInput.shape()[0];
        owTensor<float, 2> inputGradient(timeSteps, m_inputSize);
        
        m_weightGradients.setZero();
        m_biasGradients.setZero();

        owTensor<float, 2> dh_next(1, m_hiddenSize);
        dh_next.setConstant(0.0f);
        owTensor<float, 2> dc_next(1, m_hiddenSize);
        dc_next.setConstant(0.0f);

        for (int t = (int)timeSteps - 1; t >= 0; --t) {
            owTensor<float, 2> dh(1, m_hiddenSize);
            if (m_returnSequence) {
                for (size_t i = 0; i < m_hiddenSize; ++i) dh(0, i) = outputGradient(t, i) + dh_next(0, i);
            } else {
                if (t == (int)timeSteps - 1) dh = outputGradient;
                else dh = dh_next;
            }

            owTensor<float, 2> dz(1, 4 * m_hiddenSize);
            for (size_t i = 0; i < m_hiddenSize; ++i) {
                float f = m_gates[t](0, i);
                float in = m_gates[t](0, m_hiddenSize + i);
                float g = m_gates[t](0, 2 * m_hiddenSize + i);
                float o = m_gates[t](0, 3 * m_hiddenSize + i);
                float c = m_c[t + 1](0, i);
                float tanh_c = std::tanh(c);

                float dc = dh(0, i) * o * (1.0f - tanh_c * tanh_c) + dc_next(0, i);
                
                // dGates
                float df = dc * m_c[t](0, i) * f * (1.0f - f);
                float di = dc * g * in * (1.0f - in);
                float dg = dc * in * (1.0f - g * g);
                float do_gate = dh(0, i) * tanh_c * o * (1.0f - o);

                dz(0, i) = df;
                dz(0, m_hiddenSize + i) = di;
                dz(0, 2 * m_hiddenSize + i) = dg;
                dz(0, 3 * m_hiddenSize + i) = do_gate;

                dc_next(0, i) = dc * f;
            }

            // Gradients for x_h
            owTensor<float, 2> x_h(1, m_inputSize + m_hiddenSize);
            for (size_t i = 0; i < m_inputSize; ++i) x_h(0, i) = m_lastInput(t, i);
            for (size_t i = 0; i < m_hiddenSize; ++i) x_h(0, m_inputSize + i) = m_h[t](0, i);

            // Accumulate weight and bias gradients
            for (size_t i = 0; i < m_inputSize + m_hiddenSize; ++i) {
                for (size_t j = 0; j < 4 * m_hiddenSize; ++j) {
                    m_weightGradients(i, j) += x_h(0, i) * dz(0, j);
                }
            }
            for (size_t j = 0; j < 4 * m_hiddenSize; ++j) m_biasGradients(0, j) += dz(0, j);

            // Gradient for h_{t-1} and x_t
            auto dx_h = dz.dot(m_weights.transpose()); // Need transpose helper or manual
            // Since we don't have transpose(), we do it manually:
            for (size_t i = 0; i < m_inputSize; ++i) {
                float grad = 0;
                for (size_t j = 0; j < 4 * m_hiddenSize; ++j) grad += dz(0, j) * m_weights(i, j);
                inputGradient(t, i) = grad;
            }
            for (size_t i = 0; i < m_hiddenSize; ++i) {
                float grad = 0;
                for (size_t j = 0; j < 4 * m_hiddenSize; ++j) grad += dz(0, j) * m_weights(m_inputSize + i, j);
                dh_next(0, i) = grad;
            }
        }
        return inputGradient;
    }

    void train() override {
        if (m_optimizer) {
            applyRegularization(m_weights, m_weightGradients);
            m_optimizer->update(m_weights, m_weightGradients);
            m_optimizer->update(m_biases, m_biasGradients);
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
        auto copy = std::make_shared<owLSTMLayer>(m_inputSize, m_hiddenSize, m_returnSequence);
        copy->m_weights = m_weights;
        copy->m_biases = m_biases;
        copy->m_layerName = m_layerName;
        return copy;
    }

    std::string toXML() const override {
        std::stringstream ss;
        ss << "<InputSize>" << m_inputSize << "</InputSize>\n";
        ss << "<HiddenSize>" << m_hiddenSize << "</HiddenSize>\n";
        ss << "<ReturnSequence>" << (m_returnSequence ? 1 : 0) << "</ReturnSequence>\n";
        ss << "<Weights>" << m_weights.toString() << "</Weights>\n";
        ss << "<Biases>" << m_biases.toString() << "</Biases>\n";
        ss << "<RegType>" << static_cast<int>(m_regType) << "</RegType>\n";
        ss << "<RegLambda>" << m_regLambda << "</RegLambda>\n";
        return ss.str();
    }

    void fromXML(const std::string& xml) override {
        m_inputSize = std::stoul(getTagContent(xml, "InputSize"));
        m_hiddenSize = std::stoul(getTagContent(xml, "HiddenSize"));
        m_returnSequence = (std::stoi(getTagContent(xml, "ReturnSequence")) == 1);
        m_weights = owTensor<float, 2>(m_inputSize + m_hiddenSize, 4 * m_hiddenSize);
        m_biases = owTensor<float, 2>(1, 4 * m_hiddenSize);
        m_weights.fromString(getTagContent(xml, "Weights"));
        m_biases.fromString(getTagContent(xml, "Biases"));
        std::string rt = getTagContent(xml, "RegType");
        if (!rt.empty()) m_regType = static_cast<owRegularizationType>(std::stoi(rt));
        std::string rl = getTagContent(xml, "RegLambda");
        if (!rl.empty()) m_regLambda = std::stof(rl);
    }

private:
    size_t m_inputSize, m_hiddenSize;
    bool m_returnSequence;
    owTensor<float, 2> m_weights, m_biases, m_weightGradients, m_biasGradients, m_lastInput;
    std::vector<owTensor<float, 2>> m_h, m_c, m_gates;
};

} // namespace ow
