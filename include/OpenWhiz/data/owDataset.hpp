/*
 * owDataset.hpp
 *
 *  Created on: Dec 16, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <random>
#include <chrono>
#include "../core/owTensor.hpp"

/**
 * @file owDataset.hpp
 * @brief Data management and preprocessing utilities for OpenWhiz.
 */

namespace ow {

/**
 * @enum DataType
 * @brief Supported data types for dataset columns.
 */
enum class DataType { 
    Numeric,   ///< Continuous or discrete numerical values.
    Datetime,  ///< Date and time strings (to be encoded).
    Text       ///< Categorical or raw text data.
};

/**
 * @enum Ordering
 * @brief Categorical ordering strategies.
 */
enum class Ordering { 
    Standard,     ///< No special ordering.
    Categorical,  ///< Unordered categories (Nominal).
    Ordered       ///< Categories with a specific sequence (Ordinal).
};

/**
 * @enum SampleType
 * @brief Categorization for dataset splitting.
 */
enum class SampleType { 
    Training,    ///< Data used for model parameter updates.
    Validation,  ///< Data used for hyperparameter tuning and early stopping.
    Test         ///< Unseen data for final performance evaluation.
};

/**
 * @enum ImputationStrategy
 * @brief Strategies for handling missing data (NaN/Empty).
 */
enum class ImputationStrategy { 
    Mean,         ///< Replace with the column mean.
    Zero,         ///< Replace with 0.0.
    ForwardFill   ///< Replace with the previous valid value.
};

/**
 * @struct ColumnInfo
 * @brief Metadata for a single dataset column.
 */
struct ColumnInfo {
    std::string name;                          ///< Name of the column (from CSV header).
    DataType type;                             ///< Interpreted data type.
    Ordering ordering;                         ///< Categorical ordering type.
    std::map<std::string, float> category_map; ///< Mapping from category string to float ID.
    std::vector<std::string> reverse_category_map; ///< Mapping from float ID back to category string.
    float min = 0.0f;                          ///< Minimum value observed in the data.
    float max = 1.0f;                          ///< Maximum value observed in the data.
};

/**
 * @brief Internal helper to extract specific rows and columns from a tensor.
 * 
 * Used by owDataset to split the master tensor into Training, Validation, and Test sets.
 */
inline owTensor<float, 2> getRowsAndCols(const owTensor<float, 2>& full, 
                                        const std::vector<SampleType>& types, 
                                        SampleType targetType,
                                        int startCol, int numCols) {
    size_t rows = 0;
    for (auto t : types) if (t == targetType) rows++;
    if (rows == 0) return owTensor<float, 2>(0, (size_t)numCols);
    owTensor<float, 2> res(rows, (size_t)numCols);
    size_t curr = 0;
    for (size_t i = 0; i < types.size(); ++i) {
        if (types[i] == targetType) {
            for (int j = 0; j < numCols; ++j) res(curr, j) = full(i, startCol + j);
            curr++;
        }
    }
    return res;
}

/**
 * @class owDataset
 * @brief Core class for data loading, preprocessing, and management.
 * 
 * owDataset acts as the primary data provider for owNeuralNetwork. It handles 
 * reading CSV files, performing automatic Min-Max normalization, and managing 
 * the train/validation/test split.
 * 
 * Unique Features:
 * - Dynamic CSV Parsing: Automatically detects columns and handles varying delimiters.
 * - In-place Normalization: Optimizes memory usage by modifying the internal tensor directly.
 * - Reproducible Shuffling: Uses a seeded PRNG to ensure consistent splits across runs.
 * 
 * Platform Notes:
 * - Computer: Capable of handling millions of rows efficiently.
 * - Mobile/Embedded: Uses float32 to reduce memory footprint. For very large files, 
 *   external batching is recommended as owDataset loads the full file into RAM.
 */
class owDataset {
public:
    /**
     * @brief Constructs an empty dataset with default ratios (60/20/20).
     */
    owDataset() : m_targetVariableNum(1), m_autoNormalizeEnabled(false) {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        m_rng.seed(static_cast<unsigned int>(seed));
    }
    ~owDataset() = default;

    /**
     * @brief Loads and parses a CSV file.
     * @param filepath Path to the source file.
     * @param has_header If true, uses the first line as column names.
     * @return True if the file was loaded and parsed successfully.
     */
    bool loadFromCSV(const std::string& filepath, bool has_header = true) {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;
        std::string line;
        std::vector<std::vector<std::string>> raw_data;
        if (has_header && std::getline(file, line)) {
            std::stringstream ss(line);
            std::string col;
            while (std::getline(ss, col, m_delimiter)) {
                m_columns.push_back({col, DataType::Numeric, Ordering::Standard, {}, {}, 0.0f, 1.0f});
            }
        }
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string val;
            std::vector<std::string> row;
            while (std::getline(ss, val, m_delimiter)) row.push_back(val);
            if (row.empty()) continue;
            if (m_columns.empty()) {
                for (size_t i = 0; i < row.size(); ++i) m_columns.push_back({"col_" + std::to_string(i), DataType::Numeric, Ordering::Standard, {}, {}, 0.0f, 1.0f});
            }
            raw_data.push_back(row);
        }
        if (raw_data.empty()) return false;
        size_t rows = raw_data.size();
        size_t cols = m_columns.size();
        m_fullData = owTensor<float, 2>(rows, cols);
        for (size_t c = 0; c < cols; ++c) {
            for (size_t r = 0; r < rows; ++r) m_fullData(r, c) = parseValue(raw_data[r][c], m_columns[c]);
        }
        m_sampleTypes.assign(rows, SampleType::Training);
        shuffleSampleTypes();
        autoNormalize(m_autoNormalizeEnabled);
        return true;
    }

    /**
     * @brief Enables or disables automatic Min-Max scaling after data load.
     */
    void setAutoNormalizeEnabled(bool enable) { m_autoNormalizeEnabled = enable; }

    /**
     * @brief Sets the number of columns (from the right) to be treated as target variables.
     */
    void setTargetVariableNum(int num) { m_targetVariableNum = num; }

    /**
     * @brief Returns the number of target (output) variables.
     */
    int getTargetVariableNum() const { return m_targetVariableNum; }

    /**
     * @brief Returns the number of input features.
     */
    int getInputVariableNum() const { return (int)m_columns.size() - m_targetVariableNum; }

    /**
     * @brief Provides read-only access to the entire data tensor.
     */
    owTensor<float, 2> getData() const { return m_fullData; }

    /**
     * @brief Calculates min/max for each column and optionally scales values to [0, 1].
     * @param applyScaling If true, transforms the data. If false, only records min/max.
     */
    void autoNormalize(bool applyScaling = true) {
        if (m_fullData.size() == 0) return;
        size_t rows = m_fullData.shape()[0];
        size_t cols = m_fullData.shape()[1];
        for (size_t c = 0; c < cols; ++c) {
            float minVal = 1e30f, maxVal = -1e30f;
            for (size_t r = 0; r < rows; ++r) {
                minVal = std::min(minVal, m_fullData(r, c));
                maxVal = std::max(maxVal, m_fullData(r, c));
            }
            m_columns[c].min = minVal;
            m_columns[c].max = maxVal;
            if (applyScaling) {
                float range = maxVal - minVal;
                if (range == 0) range = 1.0f;
                for (size_t r = 0; r < rows; ++r) m_fullData(r, c) = (m_fullData(r, c) - minVal) / range;
            }
        }
    }

    /**
     * @brief Returns the very last sample (row) from the dataset as an input tensor.
     * @return Tensor [1, InputFeatures].
     */
    owTensor<float, 2> getLastSample() const {
        int inputSize = getInputVariableNum();
        if (m_fullData.shape()[0] == 0 || inputSize <= 0) return owTensor<float, 2>(0, 0);
        
        owTensor<float, 2> res(1, (size_t)inputSize);
        size_t lastRow = m_fullData.shape()[0] - 1;
        for (int j = 0; j < inputSize; ++j) res(0, j) = m_fullData(lastRow, j);
        return res;
    }

    /**
     * @brief Retrieves the min and max values used for normalization of a specific column.
     * @param colIdx Zero-based column index.
     * @return Pair containing {min, max}.
     */
    std::pair<float, float> getNormalizationParams(int colIdx) const {
        if (colIdx < 0 || (size_t)colIdx >= m_columns.size()) return {0.0f, 1.0f};
        return {m_columns[colIdx].min, m_columns[colIdx].max};
    }

    /**
     * @brief Returns a tensor containing the input features for the Training set.
     */
    owTensor<float, 2> getTrainInput() const { return getRowsAndCols(m_fullData, m_sampleTypes, SampleType::Training, 0, getInputVariableNum()); }

    /**
     * @brief Returns a tensor containing the target values for the Training set.
     */
    owTensor<float, 2> getTrainTarget() const { return getRowsAndCols(m_fullData, m_sampleTypes, SampleType::Training, getInputVariableNum(), m_targetVariableNum); }

    /**
     * @brief Returns a tensor containing the input features for the Validation set.
     */
    owTensor<float, 2> getValInput() const { return getRowsAndCols(m_fullData, m_sampleTypes, SampleType::Validation, 0, getInputVariableNum()); }

    /**
     * @brief Returns a tensor containing the target values for the Validation set.
     */
    owTensor<float, 2> getValTarget() const { return getRowsAndCols(m_fullData, m_sampleTypes, SampleType::Validation, getInputVariableNum(), m_targetVariableNum); }

    /**
     * @brief Returns a tensor containing the input features for the Test set.
     */
    owTensor<float, 2> getTestInput() const { return getRowsAndCols(m_fullData, m_sampleTypes, SampleType::Test, 0, getInputVariableNum()); }

    /**
     * @brief Returns a tensor containing the target values for the Test set.
     */
    owTensor<float, 2> getTestTarget() const { return getRowsAndCols(m_fullData, m_sampleTypes, SampleType::Test, getInputVariableNum(), m_targetVariableNum); }

    /**
     * @brief Configures the ratios for splitting the data.
     * @param train Ratio for training (e.g., 0.8).
     * @param val Ratio for validation (e.g., 0.1).
     * @param test Ratio for testing (e.g., 0.1).
     */
    void setRatios(float train, float val, float test) {
        m_trainRatio = train; m_valRatio = val; m_testRatio = test;
        shuffleSampleTypes();
    }

    /**
     * @brief Sets the character used to separate values in the CSV file.
     * @param d The delimiter character (e.g., ',', ';', '\t').
     */
    void setDelimiter(char d) { m_delimiter = d; }

    /**
     * @brief Transforms the tabular data into a sliding window format for forecasting.
     * 
     * Concatenates features from 'windowSize' consecutive steps to form a single input 
     * vector. The target is the value of the 'targetIdx' column at the immediate next step.
     * 
     * @param windowSize Number of time steps in each input window.
     * @param dilation Gap between steps within a window (1 = consecutive, 2 = every other step).
     * @param targetIdx Column index to be used as the prediction target. If -1, uses the last column.
     */
    void prepareForecastingData(int windowSize, int dilation = 1, int targetIdx = -1) {
        if (m_fullData.size() == 0) return;
        size_t originalRows = m_fullData.shape()[0];
        size_t originalCols = m_fullData.shape()[1];
        
        // Safety check: total span of window must fit within the dataset
        if ((size_t)(windowSize * dilation) >= originalRows) return;

        // Internal horizon is fixed to 1 for recursive forecasting support
        int horizon = 1;
        
        // Handle default target index (last column)
        if (targetIdx == -1) targetIdx = (int)originalCols - 1;
        if (targetIdx < 0 || (size_t)targetIdx >= originalCols) return;

        int lastWindowIndexOffset = (windowSize - 1) * dilation;
        int targetIndexOffset = lastWindowIndexOffset + horizon;

        if ((size_t)targetIndexOffset >= originalRows) return;
        
        size_t newRows = originalRows - targetIndexOffset;
        size_t newInputCols = (size_t)windowSize * originalCols;
        size_t newTargetCols = 1;
        
        owTensor<float, 2> newFullData(newRows, newInputCols + newTargetCols);
        
        for (size_t i = 0; i < newRows; ++i) {
            // Fill input (dilated window)
            for (int w = 0; w < windowSize; ++w) {
                int rowIdx = i + w * dilation;
                for (size_t c = 0; c < originalCols; ++c) {
                    newFullData(i, w * originalCols + c) = m_fullData(rowIdx, c);
                }
            }
            // Fill target
            newFullData(i, newInputCols) = m_fullData(i + targetIndexOffset, (size_t)targetIdx);
        }
        
        m_fullData = newFullData;
        
        // Update columns metadata
        std::vector<ColumnInfo> newColumns;
        for (int w = 0; w < windowSize; ++w) {
            int t_offset = (windowSize - 1 - w) * dilation;
            for (size_t c = 0; c < originalCols; ++c) {
                newColumns.push_back({m_columns[c].name + "_t-" + std::to_string(t_offset), 
                                      m_columns[c].type, m_columns[c].ordering, {}, {}, 0.0f, 1.0f});
            }
        }
        newColumns.push_back({m_columns[targetIdx].name + "_target", 
                              m_columns[targetIdx].type, m_columns[targetIdx].ordering, {}, {}, 0.0f, 1.0f});
        
        m_columns = newColumns;
        m_targetVariableNum = 1;
        
        m_sampleTypes.assign(newRows, SampleType::Training);
        shuffleSampleTypes();
        autoNormalize(m_autoNormalizeEnabled);
    }

    /**
     * @brief Randomly assigns each row to Training, Validation, or Test groups.
     */
    void shuffleSampleTypes() {
        if (m_sampleTypes.empty()) return;
        size_t rows = m_sampleTypes.size();
        size_t trainCount = (size_t)(rows * m_trainRatio);
        size_t valCount = (size_t)(rows * m_valRatio);
        std::vector<SampleType> newTypes(rows);
        for (size_t i = 0; i < rows; ++i) {
            if (i < trainCount) newTypes[i] = SampleType::Training;
            else if (i < trainCount + valCount) newTypes[i] = SampleType::Validation;
            else newTypes[i] = SampleType::Test;
        }
        std::shuffle(newTypes.begin(), newTypes.end(), m_rng);
        m_sampleTypes = newTypes;
    }

    /**
     * @brief Returns a string representation of the sample type for a given row.
     */
    std::string getSampleTypeString(size_t index) const {
        if (index >= m_sampleTypes.size()) return "Unknown";
        if (m_sampleTypes[index] == SampleType::Training) return "Training";
        if (m_sampleTypes[index] == SampleType::Validation) return "Validation";
        return "Testing";
    }

private:
    owTensor<float, 2> m_fullData;      ///< Master tensor holding all observations.
    std::vector<ColumnInfo> m_columns; ///< Metadata for each column.
    std::vector<SampleType> m_sampleTypes; ///< Allocation of each row to a split.
    int m_targetVariableNum = 1;
    float m_trainRatio = 0.6f, m_valRatio = 0.2f, m_testRatio = 0.2f;
    char m_delimiter = ';';
    bool m_autoNormalizeEnabled = false;
    std::mt19937 m_rng;

    /**
     * @brief Parses a string value into a float.
     * 
     * Handles basic numeric conversion. Future extensions will include 
     * categorical mapping and datetime encoding.
     */
    float parseValue(const std::string& val, ColumnInfo& info) {
        try { return std::stof(val); } catch (...) { return 0.0f; }
    }
};

} // namespace ow
