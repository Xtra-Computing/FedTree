//
// Created by hanyuxuan on 28/10/20.
//

#include "gtest/gtest.h"
#include "FedTree/Tree/tree_builder.h"

class TreeBuilderTest : public ::testing::Test {
public:
    vector<float> gradients;
    vector<int> splits;
protected:
    void SetUp() override {
        gradients = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
        splits = {1, 4};
    }
};

TEST_F(TreeBuilderTest, compute_histogram) {
    printf("### Test compute_histogram ###\n");
    vector<float> histogram = TreeBuilder().compute_histogram(gradients, splits);
    EXPECT_EQ(histogram.size(), splits.size() + 1);
    EXPECT_NEAR(histogram[0], 0.1, 1e-5);
    EXPECT_NEAR(histogram[1], 0.3, 1e-5);
    EXPECT_NEAR(histogram[2], 0.4, 1e-5);
}