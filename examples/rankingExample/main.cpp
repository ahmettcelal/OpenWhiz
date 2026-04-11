#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <iomanip>
#include "OpenWhiz/openwhiz.hpp"

/**
 * @brief Synthetic ranking data generator.
 * Creates pairs of items (each with 3 features) and a ranking label.
 * Scoring rule: score = 2.0*f1 + 1.0*f2 - 0.5*f3.
 * Label: 1 if score1 > score2, -1 otherwise.
 */
void generateRankingData(const std::string& filename, int count) {
    std::ofstream file(filename);
    file << "x1_f1;x1_f2;x1_f3;x2_f1;x2_f2;x2_f3;RankLabel\n";
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    auto getScore = [](float f1, float f2, float f3) {
        return 2.0f * f1 + 1.0f * f2 - 0.5f * f3;
    };

    for (int i = 0; i < count; ++i) {
        float f1_1 = dist(gen), f2_1 = dist(gen), f3_1 = dist(gen);
        float f1_2 = dist(gen), f2_2 = dist(gen), f3_2 = dist(gen);
        
        float s1 = getScore(f1_1, f2_1, f3_1);
        float s2 = getScore(f1_2, f2_2, f3_2);
        
        // Label is 1 if item1 is better, -1 if item2 is better
        float label = (s1 > s2) ? 1.0f : -1.0f;

        file << f1_1 << ";" << f2_1 << ";" << f3_1 << ";"
             << f1_2 << ";" << f2_2 << ";" << f3_2 << ";" << label << "\n";
    }
}

int main() {
    std::cout << "=== OpenWhiz Pairwise Ranking Example ===\n" << std::endl;

    const std::string csvFile = "ranking_data.csv";
    generateRankingData(csvFile, 1000);

    ow::owNeuralNetwork nn;
    
    // 1. Setup Data
    if (!nn.loadData(csvFile)) {
        std::cerr << "Failed to load data!" << std::endl;
        return -1;
    }
    nn.getDataset()->setTargetVariableNum(1); // Label is -1 or 1
    nn.getDataset()->setRatios(0.8f, 0.1f, 0.1f);

    // 2. Build Ranking Architecture
    // Input is concatenated [x1, x2] (3+3 = 6 dimensions)
    auto rankingLayer = std::make_shared<ow::owRankingLayer>(3);
    nn.addLayer(rankingLayer);

    // 3. Configure Ranking Training
    // Use Margin Ranking Loss with margin 1.0
    nn.setLoss(std::make_shared<ow::owMarginRankingLoss>(1.0f));
    nn.setOptimizer(std::make_shared<ow::owADAMOptimizer>(0.01f));
    nn.setMaximumEpochNum(200);
    nn.setPrintEpochInterval(20);

    // 4. Train
    std::cout << "Training model to rank items correctly..." << std::endl;
    nn.train();

    // 5. Final Evaluation
    // For ranking, "Accuracy" means "how often the higher scored item was the winner"
    auto testIn = nn.getDataset()->getTestInput();
    auto testOut = nn.getDataset()->getTestTarget();
    auto predictions = nn.forward(testIn);
    
    int correctCount = 0;
    for(size_t i=0; i<testIn.shape()[0]; ++i) {
        float s1 = predictions(i, 0);
        float s2 = predictions(i, 1);
        float label = testOut(i, 0);
        
        if ((s1 > s2 && label > 0) || (s1 < s2 && label < 0)) {
            correctCount++;
        }
    }
    
    std::cout << "\nRanking Accuracy on Test Set: " 
              << (100.0f * correctCount / testIn.shape()[0]) << "%" << std::endl;

    // 6. Inspect learned weights
    // Our true weights were [2.0, 1.0, -0.5]
    std::cout << "\nLearned Ranking Weights (Feature Importance):" << std::endl;
    std::cout << rankingLayer->toXML() << std::endl;

    // 7. Test a new pair
    // Pair: [5, 5, 0] vs [0, 0, 0] -> Clearly [5,5,0] should be higher
    ow::owTensor<float, 2> pair(1, 6);
    pair(0, 0) = 5.0f; pair(0, 1) = 5.0f; pair(0, 2) = 0.0f; // Item 1
    pair(0, 3) = 0.0f; pair(0, 4) = 0.0f; pair(0, 5) = 0.0f; // Item 2
    
    auto scores = nn.forward(pair);
    std::cout << "Scores for Pair ([5,5,0] vs [0,0,0]):" << std::endl;
    std::cout << "  Item 1 Score: " << scores(0, 0) << std::endl;
    std::cout << "  Item 2 Score: " << scores(0, 1) << std::endl;
    std::cout << "Result: " << (scores(0, 0) > scores(0, 1) ? "Item 1 ranks higher!" : "Item 2 ranks higher!") << std::endl;

    return 0;
}
