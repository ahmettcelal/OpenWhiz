/*
 * owConcatenateLayer.hpp
 *
 *  Created on: Apr 10, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <vector>
#include <memory>
#include <numeric>

namespace ow {

/**
 * @class owConcatenateLayer
 * @brief A meta-layer that executes multiple branches in parallel and concatenates their outputs along the feature dimension.
 * 
 * This layer acts as a branching point in the network. It splits a single input tensor into multiple parts,
 * passes each part through a dedicated "branch" (another owLayer), and then merges the results back into 
 * a single wide tensor.
 * 
 * @details
 * **Workflow:**
 * 1. **Input Slicing:** The input [Batch, TotalInputSize] is sliced horizontally. Each branch `i` 
 *    receives a slice of width `branch[i]->getInputSize()`.
 * 2. **Parallel Execution:** Each branch executes its `forward` pass independently.
 * 3. **Output Concatenation:** The results [Batch, OutputSize_i] are concatenated into a single result 
 *    tensor [Batch, Sum(OutputSize_i)].
 * 
 * **Use Cases:**
 * - Multi-scale feature extraction (e.g., parallel windows in time-series).
 * - Combining different types of input encodings.
 * - Implementing Inception-like architectural blocks.
 */
class owConcatenateLayer : public owLayer {
public:
    /**
     * @brief Constructs an owConcatenateLayer with an optional initial set of branches.
     * @param branches A vector of shared pointers to the layers that will form the parallel branches.
     */
    owConcatenateLayer(const std::vector<std::shared_ptr<owLayer>>& branches = {})
        : m_branches(branches) {
        m_layerName = "Concatenate Layer";
    }

    /**
     * @brief Adds a new parallel branch to the layer.
     * @param branch Shared pointer to the layer to be added as a branch.
     */
    void addBranch(std::shared_ptr<owLayer> branch) {
        if (branch) m_branches.push_back(branch);
    }

    /**
     * @brief Replaces the current branches with a new set.
     * @details This is primarily used during XML deserialization to reconstruct complex architectures.
     * @param branches Vector of shared pointers to the new layers.
     */
    void setBranches(const std::vector<std::shared_ptr<owLayer>>& branches) {
        m_branches = branches;
    }

    /**
     * @brief Calculates the total input size expected by this layer.
     * @return The sum of the input sizes of all branches.
     */
    size_t getInputSize() const override {
        size_t total = 0;
        for (const auto& b : m_branches) total += b->getInputSize();
        return total;
    }

    /**
     * @brief Calculates the total output size produced by this layer.
     * @return The sum of the output sizes of all branches.
     */
    size_t getOutputSize() const override {
        size_t total = 0;
        for (const auto& b : m_branches) total += b->getOutputSize();
        return total;
    }

    /**
     * @brief Implementation of virtual setNeuronNum. 
     * @note Concatenate layer's "neurons" are determined by its branches; this call is ignored.
     */
    void setNeuronNum(size_t num) override { (void)num; }

    /**
     * @brief Performs the forward pass by slicing the input and executing all branches.
     * @param input Input tensor of shape [BatchSize, TotalInputSize].
     * @return Concatenated output tensor of shape [BatchSize, TotalOutputSize].
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        size_t batch = input.shape()[0];
        m_outputs.clear();
        m_outputs.reserve(m_branches.size());

        size_t currentInOffset = 0;
        for (auto& branch : m_branches) {
            size_t inSize = branch->getInputSize();
            
            // Slice input for this branch
            owTensor<float, 2> slicedInput(batch, inSize);
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < inSize; ++j) {
                    slicedInput(i, j) = input(i, currentInOffset + j);
                }
            }
            
            m_outputs.push_back(branch->forward(slicedInput));
            currentInOffset += inSize;
        }

        // Concatenate all branch outputs
        owTensor<float, 2> result(batch, getOutputSize());
        size_t currentOutOffset = 0;
        for (const auto& out : m_outputs) {
            size_t outSize = out.shape()[1];
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < outSize; ++j) {
                    result(i, currentOutOffset + j) = out(i, j);
                }
            }
            currentOutOffset += outSize;
        }

        return result;
    }

    /**
     * @brief Performs the backward pass by splitting the gradient and propagating through branches.
     * @param outputGradient Gradient from the next layer [BatchSize, TotalOutputSize].
     * @return Combined input gradient [BatchSize, TotalInputSize].
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        size_t batch = outputGradient.shape()[0];
        std::vector<owTensor<float, 2>> inputGradients;
        inputGradients.reserve(m_branches.size());

        size_t currentOutOffset = 0;
        for (size_t k = 0; k < m_branches.size(); ++k) {
            size_t outSize = m_outputs[k].shape()[1];
            
            // Slice output gradient
            owTensor<float, 2> slicedGrad(batch, outSize);
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < outSize; ++j) {
                    slicedGrad(i, j) = outputGradient(i, currentOutOffset + j);
                }
            }
            
            inputGradients.push_back(m_branches[k]->backward(slicedGrad));
            currentOutOffset += outSize;
        }

        // Concatenate all input gradients
        owTensor<float, 2> result(batch, getInputSize());
        size_t currentInOffset = 0;
        for (const auto& inGrad : inputGradients) {
            size_t inSize = inGrad.shape()[1];
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < inSize; ++j) {
                    result(i, currentInOffset + j) = inGrad(i, j);
                }
            }
            currentInOffset += inSize;
        }

        return result;
    }

    /**
     * @brief Triggers the training update for all internal branches.
     */
    void train() override {
        for (auto& branch : m_branches) branch->train();
    }

    /**
     * @brief Resets all internal branches.
     */
    void reset() override {
        for (auto& branch : m_branches) branch->reset();
    }

    /**
     * @brief Creates a deep copy of the concatenate layer and all its branches.

     * @return Shared pointer to the cloned layer.
     */
    std::shared_ptr<owLayer> clone() const override {
        std::vector<std::shared_ptr<owLayer>> branchCopies;
        for (const auto& b : m_branches) branchCopies.push_back(b->clone());
        auto copy = std::make_shared<owConcatenateLayer>(branchCopies);
        copy->m_layerName = m_layerName;
        return copy;
    }

    /**
     * @brief Serializes the layer and its recursive branch structure to XML.
     * @return XML string representing the layer.
     */
    std::string toXML() const override {
        std::stringstream ss;
        ss << "<BranchCount>" << m_branches.size() << "</BranchCount>\n";
        for (size_t i = 0; i < m_branches.size(); ++i) {
            ss << "<Branch_" << i << " type=\"" << m_branches[i]->getLayerName() << "\">\n" 
               << m_branches[i]->toXML() << "</Branch_" << i << ">\n";
        }
        return ss.str();
    }

    /**
     * @brief Deserializes the branch parameters from XML.
     * @note Structural reconstruction is handled by owNeuralNetwork::loadFromXML.
     * @param xml XML content for the layer.
     */
    void fromXML(const std::string& xml) override {
        // Full reconstruction is handled recursively by owNeuralNetwork::loadFromXML.
        // This method will only update parameters if the branches are already set.
        for (size_t i = 0; i < m_branches.size(); ++i) {
            std::string tag = "Branch_" + std::to_string(i);
            m_branches[i]->fromXML(owLayer::getNestedTagContent(xml, tag));
        }
    }

    /** @return Always nullptr as this is a meta-layer. */
    float* getParamsPtr() override { return nullptr; }
    /** @return Always nullptr as this is a meta-layer. */
    float* getGradsPtr() override { return nullptr; }
    /** @return Always 0 as this layer only delegates to its branches. */
    size_t getParamsCount() override { return 0; }

private:
    std::vector<std::shared_ptr<owLayer>> m_branches; ///< Internal storage for parallel branches.
    std::vector<owTensor<float, 2>> m_outputs;      ///< Cached outputs of branches for the backward pass.
};

} // namespace ow
