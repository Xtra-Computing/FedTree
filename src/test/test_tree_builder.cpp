//
// Created by hanyuxuan on 28/10/20.
//

#include "gtest/gtest.h"
#include "FedTree/Tree/tree_builder.h"
#include "FedTree/Tree/hist_tree_builder.h"
#include "FedTree/Tree/hist_cut.h"
#include <string>

class TreeBuilderTest : public ::testing::Test {
public:

protected:
    void SetUp() override {
    }
};

TEST_F(TreeBuilderTest, compute_histogram) {
    printf("### Test compute_histogram ###\n");
    int n_instances = 4;
    int n_columns = 2;

    SyncArray<GHPair> gradients(4);
    const vector<GHPair> gradients_vec = {GHPair(0.1, 0.2), GHPair(0.3, 0.4), GHPair(0.5, 0.6), GHPair(0.7, 0.8)};
    gradients.copy_from(&gradients_vec[0], gradients_vec.size());

    HistCut cut;
    cut.cut_row_ptr = SyncArray<int>(3);
    const vector<int> cut_row_ptr_vec = {0, 1, 3};
    cut.cut_row_ptr.copy_from(&cut_row_ptr_vec[0], cut_row_ptr_vec.size());

    SyncArray<unsigned char> dense_bin_id(8);
    const vector<unsigned char> bin_vec = {0, 0, 0, 1, 1, 1, 1, 2};
    dense_bin_id.copy_from(&bin_vec[0], bin_vec.size());

    SyncArray<GHPair> hist(5);
    HistTreeBuilder htb;
    hist.copy_from(htb.compute_histogram(n_instances, n_columns, gradients, cut, dense_bin_id));
    auto hist_data = hist.host_data();
    EXPECT_NEAR(hist_data[0].g, 0.4, 1e-5);
    EXPECT_NEAR(hist_data[0].h, 0.6, 1e-5);
    EXPECT_NEAR(hist_data[1].g, 1.2, 1e-5);
    EXPECT_NEAR(hist_data[1].h, 1.4, 1e-5);
    EXPECT_NEAR(hist_data[2].g, 0.1, 1e-5);
    EXPECT_NEAR(hist_data[2].h, 0.2, 1e-5);
    EXPECT_NEAR(hist_data[3].g, 0.8, 1e-5);
    EXPECT_NEAR(hist_data[3].h, 1.0, 1e-5);
    EXPECT_NEAR(hist_data[4].g, 0.7, 1e-5);
    EXPECT_NEAR(hist_data[4].h, 0.8, 1e-5);

//    vector<float> histogram = TreeBuilder().compute_histogram(gradients, splits);
//    EXPECT_EQ(histogram.size(), splits.size() + 1);
//    EXPECT_NEAR(histogram[0], 0.1, 1e-5);
//    EXPECT_NEAR(histogram[1], 0.3, 1e-5);
//    EXPECT_NEAR(histogram[2], 0.4, 1e-5);
}