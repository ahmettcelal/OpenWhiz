/*
 * owStatistics.hpp
 *
 *  Created on: Mar 12, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include <cmath>
#include <numeric>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include "owDataset.hpp"

/**
 * @file owStatistics.hpp
 * @brief Advanced statistical analysis tools for OpenWhiz datasets.
 */

namespace ow {

/**
 * @struct StatisticsReport
 * @brief Comprehensive report for statistical analysis of a dataset.
 * 
 * Provides insights into the quality and suitability of data for regression models.
 */
struct StatisticsReport {
    float rSquared = 0.0f;              ///< Coefficient of determination.
    float adjRSquared = 0.0f;           ///< Adjusted R-Squared for model complexity.
    float durbinWatson = 0.0f;          ///< Measures autocorrelation in residuals (Target: 1.5 - 2.5).
    float skewness = 0.0f;              ///< Measures symmetry of the distribution.
    float kurtosis = 0.0f;              ///< Measures "peakedness" of the distribution.
    float homoscedasticityScore = 0.0f; ///< Measure of variance consistency across residuals.
    std::vector<float> cooksDistance;   ///< List of values representing the influence of each data point.
    bool isSuitable = false;            ///< Final recommendation flag.
    std::string recommendation;         ///< Human-readable advice based on the analysis.
};

/**
 * @struct TTestResult
 * @brief Result structure for a Student's T-Test.
 */
struct TTestResult {
    float tValue = 0.0f;       ///< Calculated T-statistic.
    float pValue = 0.0f;       ///< Probability value (significance).
    int df = 0;                ///< Degrees of freedom.
    bool isSignificant = false; ///< True if p < 0.05 (significant difference).
};

/**
 * @struct ANOVAResult
 * @brief Result structure for a One-Way ANOVA.
 */
struct ANOVAResult {
    float fValue = 0.0f;        ///< Calculated F-statistic.
    float pValue = 0.0f;        ///< Probability value.
    float ssBetween = 0.0f;     ///< Sum of Squares Between Groups.
    float ssWithin = 0.0f;      ///< Sum of Squares Within Groups (Error).
    int dfBetween = 0;          ///< Degrees of freedom (Between).
    int dfWithin = 0;           ///< Degrees of freedom (Within).
    bool isSignificant = false; ///< True if p < 0.05.
};

/**
 * @class owDatasetStatistics
 * @brief Engine for performing complex statistical tests on owDataset.
 * 
 * This class provides a suite of tools to validate data assumptions before 
 * training neural networks. It helps in identifying multicollinearity, 
 * outliers, and non-linear patterns.
 * 
 * Unique Features:
 * - Suitability Logic: Recommends specific layer types (e.g., LSTM) based on autocorrelation.
 * - Influence Detection: Uses Cook's Distance to find samples that might skew model training.
 * - Multi-group Analysis: Supports ANOVA for checking variance across multiple feature sets.
 * 
 * Platform Notes:
 * - Computer/Industrial: Recommended for "Pre-flight" data checks before large training jobs.
 * - Mobile/Web: Low overhead; can be used for on-device data validation.
 */
class owDatasetStatistics {
public:
    /**
     * @brief Constructs the statistics engine.
     */
    owDatasetStatistics() : m_dataset(nullptr) {}

    /**
     * @brief Connects an existing owDataset to the engine.
     * @param ds Pointer to the dataset to analyze.
     */
    void setDataset(owDataset* ds) { m_dataset = ds; }

    /**
     * @brief Calculates the Pearson Correlation Coefficient between two columns.
     * @param col1 First column index.
     * @param col2 Second column index.
     * @return Float between -1.0 and 1.0.
     */
    float calculateCorrelation(int col1, int col2) {
        if (!m_dataset) return 0.0f;
        auto data = m_dataset->getData();
        if (data.size() == 0) return 0.0f;
        
        size_t n = data.shape()[0];
        float sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, pSum = 0;
        
        for (size_t i = 0; i < n; ++i) {
            float x = data(i, col1);
            float y = data(i, col2);
            sum1 += x;
            sum2 += y;
            sum1Sq += x * x;
            sum2Sq += y * y;
            pSum += x * y;
        }

        float num = pSum - (sum1 * sum2 / static_cast<float>(n));
        float den = std::sqrt((sum1Sq - sum1 * sum1 / static_cast<float>(n)) * 
                              (sum2Sq - sum2 * sum2 / static_cast<float>(n)));

        return (den == 0) ? 0.0f : num / den;
    }

    /**
     * @brief Performs a deep Ordinary Least Squares (OLS) regression analysis.
     * 
     * Analyzes linear relationship, residuals, and influence points.
     * @param xCol Input feature column index.
     * @param yCol Target variable column index.
     * @return A StatisticsReport with suitability advice.
     */
    StatisticsReport analyzeRegressionSuitability(int xCol, int yCol) {
        StatisticsReport report;
        if (!m_dataset) return report;
        
        auto data = m_dataset->getData();
        size_t n = data.shape()[0];
        if (n < 10) {
            report.isSuitable = false;
            report.recommendation = "Sample size too small for reliable statistics.";
            return report;
        }

        // 1. OLS Regression (y = ax + b)
        float sX = 0, sY = 0, sXY = 0, sXX = 0, sYY = 0;
        for (size_t i = 0; i < n; ++i) {
            float x = data(i, xCol), y = data(i, yCol);
            sX += x; sY += y; sXY += x * y; sXX += x * x; sYY += y * y;
        }
        float denominator = (n * sXX - sX * sX);
        float a = (denominator == 0) ? 0 : (n * sXY - sX * sY) / denominator;
        float b = (sY - a * sX) / n;

        // 2. Residuals and R-Squared
        std::vector<float> residuals(n);
        float ssRes = 0, ssTot = 0, meanY = sY / n;
        for (size_t i = 0; i < n; ++i) {
            float pred = a * data(i, xCol) + b;
            residuals[i] = data(i, yCol) - pred;
            ssRes += residuals[i] * residuals[i];
            ssTot += (data(i, yCol) - meanY) * (data(i, yCol) - meanY);
        }
        report.rSquared = (ssTot == 0) ? 0 : 1.0f - (ssRes / ssTot);
        report.adjRSquared = 1.0f - (1.0f - report.rSquared) * (n - 1) / (n - 2);

        // 3. Autocorrelation (Durbin-Watson)
        float dwNum = 0;
        for (size_t i = 1; i < n; ++i) dwNum += std::pow(residuals[i] - residuals[i-1], 2);
        report.durbinWatson = (ssRes == 0) ? 0 : dwNum / ssRes;

        // 4. Cook's Distance (Identifying influential outliers)
        float mse = ssRes / (n - 2);
        float meanX = sX / n;
        float ssX = sXX - (sX * sX / n);
        report.cooksDistance.resize(n);
        for (size_t i = 0; i < n; ++i) {
            float leverage = (1.0f / n) + std::pow(data(i, xCol) - meanX, 2) / ssX;
            if (leverage >= 1.0f) leverage = 0.999f;
            report.cooksDistance[i] = (std::pow(residuals[i], 2) / (2 * mse)) * (leverage / std::pow(1 - leverage, 2));
        }

        // 5. Normality (Skewness/Kurtosis)
        float m3 = 0, m4 = 0, s2 = ssRes / n;
        for (float r : residuals) {
            m3 += std::pow(r, 3);
            m4 += std::pow(r, 4);
        }
        report.skewness = (n * m3) / ((n - 1) * (n - 2) * std::pow(s2, 1.5f));
        report.kurtosis = (n * (n + 1) * m4) / ((n - 1) * (n - 2) * (n - 3) * std::pow(s2, 2));

        // Suitability Logic
        bool goodR2 = report.rSquared > 0.6f;
        bool noAutoCorr = report.durbinWatson > 1.2f && report.durbinWatson < 2.8f;
        bool lowOutliers = *std::max_element(report.cooksDistance.begin(), report.cooksDistance.end()) < 1.0f;

        report.isSuitable = goodR2 && noAutoCorr && lowOutliers;
        
        if (report.isSuitable) report.recommendation = "Data shows strong linear patterns. Regression is highly recommended.";
        else if (!goodR2) report.recommendation = "Low correlation detected. Consider a non-linear model (Neural Network).";
        else if (!noAutoCorr) report.recommendation = "High autocorrelation. Use time-series layers (LSTM).";
        else report.recommendation = "High influence points (outliers) detected. Check Cook's Distance.";

        return report;
    }

    /**
     * @brief Performs a Student's T-Test between two columns.
     * 
     * Checks if the means of two populations are significantly different.
     * @param col1 First column index.
     * @param col2 Second column index.
     * @return TTestResult with significance flag.
     */
    TTestResult performTTest(int col1, int col2) {
        TTestResult result;
        if (!m_dataset) return result;
        auto data = m_dataset->getData();
        size_t n = data.shape()[0];
        if (n < 2) return result;

        float sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0;
        for (size_t i = 0; i < n; ++i) {
            float v1 = data(i, col1), v2 = data(i, col2);
            sum1 += v1; sum2 += v2;
            sum1Sq += v1 * v1; sum2Sq += v2 * v2;
        }

        float mean1 = sum1 / n, mean2 = sum2 / n;
        float var1 = (sum1Sq - (sum1 * sum1) / n) / (n - 1);
        float var2 = (sum2Sq - (sum2 * sum2) / n) / (n - 1);

        result.tValue = (mean1 - mean2) / std::sqrt((var1 / n) + (var2 / n));
        result.df = (int)(2 * n - 2);
        // Simple p-value approximation for Alpha = 0.05
        result.isSignificant = std::abs(result.tValue) > 2.0f; 
        return result;
    }

    /**
     * @brief Performs a One-Way Analysis of Variance (ANOVA).
     * 
     * Tests if the means of three or more groups are different.
     * @param columns List of column indices representing different groups.
     * @return ANOVAResult.
     */
    ANOVAResult performOneWayANOVA(const std::vector<int>& columns) {
        ANOVAResult result;
        if (!m_dataset || columns.size() < 2) return result;
        auto data = m_dataset->getData();
        size_t n = data.shape()[0];
        size_t k = columns.size();

        float grandSum = 0;
        std::vector<float> groupMeans(k);
        for (size_t j = 0; j < k; ++j) {
            float groupSum = 0;
            for (size_t i = 0; i < n; ++i) groupSum += data(i, columns[j]);
            groupMeans[j] = groupSum / n;
            grandSum += groupSum;
        }
        float grandMean = grandSum / (n * k);

        for (size_t j = 0; j < k; ++j) {
            result.ssBetween += n * std::pow(groupMeans[j] - grandMean, 2);
            for (size_t i = 0; i < n; ++i) {
                result.ssWithin += std::pow(data(i, columns[j]) - groupMeans[j], 2);
            }
        }

        result.dfBetween = (int)(k - 1);
        result.dfWithin = (int)(k * (n - 1));
        float msBetween = result.ssBetween / result.dfBetween;
        float msWithin = result.ssWithin / result.dfWithin;
        result.fValue = (msWithin == 0) ? 0 : msBetween / msWithin;
        result.isSignificant = result.fValue > 3.0f; // Simple proxy for Alpha = 0.05
        return result;
    }

    /**
     * @brief Calculates Variance Inflation Factor (VIF).
     * 
     * Detects multicollinearity (redundancy) among features.
     * @param targetCol The feature column to check against all others.
     * @return Float value (VIF > 5 or 10 indicates high redundancy).
     */
    float calculateVIF(int targetCol) {
        if (!m_dataset) return 0.0f;
        auto data = m_dataset->getData();
        size_t n = data.shape()[0];
        size_t cols = data.shape()[1];
        if (cols < 2) return 1.0f;

        // Simplified VIF using Mean Correlation as proxy for R-Squared
        float sumCorrSq = 0;
        int count = 0;
        for (size_t i = 0; i < cols; ++i) {
            if ((int)i == targetCol) continue;
            float r = calculateCorrelation(targetCol, (int)i);
            sumCorrSq += r * r;
            count++;
        }
        float avgRSq = sumCorrSq / count;
        if (avgRSq >= 1.0f) return 99.9f;
        return 1.0f / (1.0f - avgRSq);
    }

    /**
     * @brief Performs Chi-Squared Test of Independence.
     * 
     * Used for categorical columns to see if they are related.
     * @param col1 First categorical column.
     * @param col2 Second categorical column.
     * @return Chi-Square statistic.
     */
    float performChiSquaredTest(int col1, int col2) {
        if (!m_dataset) return 0.0f;
        auto data = m_dataset->getData();
        size_t n = data.shape()[0];

        // Find unique categories
        std::map<float, int> cat1, cat2;
        for (size_t i = 0; i < n; ++i) {
            cat1[data(i, col1)]++;
            cat2[data(i, col2)]++;
        }

        // Observed Frequencies
        std::map<std::pair<float, float>, int> observed;
        for (size_t i = 0; i < n; ++i) observed[{data(i, col1), data(i, col2)}]++;

        float chiSq = 0;
        for (auto const& entry1 : cat1) {
            float v1 = entry1.first;
            int count1 = entry1.second;
            for (auto const& entry2 : cat2) {
                float v2 = entry2.first;
                int count2 = entry2.second;
                float expected = (static_cast<float>(count1) * count2) / n;
                float obs = static_cast<float>(observed[{v1, v2}]);
                if (expected > 0) chiSq += std::pow(obs - expected, 2) / expected;
            }
        }
        return chiSq;
    }

private:
    owDataset* m_dataset; ///< Pointer to the source dataset.
};

} // namespace ow
