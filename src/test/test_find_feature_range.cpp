//
// Created by Kelly Yung on 2021/2/25.
//

#include "FedTree/FL/party.h"
#include "gtest/gtest.h"
#include <vector>


class FeatureRangeTest: public ::testing::Test {
public:

    vector<float_type> csc_val = {10, 20, 30, 50, 40, 60, 70, 80};
    vector<int> csc_row_idx = {0, 0, 1, 1, 2, 2, 2, 4};
    vector<int> csc_col_ptr = {0, 1, 3, 4, 6, 7, 8};
    Party p;

protected:
    void SetUp() override {
        p.dataset.csc_row_idx = csc_row_idx;
        p.dataset.csc_col_ptr = csc_col_ptr;
        p.dataset.csc_val = csc_val;
    }
};

TEST_F(FeatureRangeTest, find_feature_range_by_index_single_value){
    vector<float> result = p.get_feature_range_by_feature_index(0);
    LOG(INFO) << "Result:" << result;
    EXPECT_EQ(result[0], 10);
    EXPECT_EQ(result[1], 10);
}

TEST_F(FeatureRangeTest, find_feature_range_by_index_multi_value){
    vector<float> result = p.get_feature_range_by_feature_index(1);
    EXPECT_EQ(result[0], 20);
    EXPECT_EQ(result[1], 30);
}

TEST_F(FeatureRangeTest, find_feature_range_by_last_index){
vector<float> result = p.get_feature_range_by_feature_index(5);
EXPECT_EQ(result[0], 80);
EXPECT_EQ(result[1], 80);
}
