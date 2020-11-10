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

TEST_F(TreeBuilderTest, merge_histogram) {
    printf("### Test merge_histogram ###\n");
    int n_bins = 3;

    const vector<GHPair> hist1_vec = {GHPair(0.1, 0.2), GHPair(0.3, 0.4), GHPair(0.5, 0.6)};
    const vector<GHPair> hist2_vec = {GHPair(0.11, 0.22), GHPair(0.33, 0.44), GHPair(0.55, 0.66)};

    MSyncArray<GHPair> hists(2, n_bins);
    hists[0].copy_from(&hist1_vec[0], hist1_vec.size());
    hists[1].copy_from(&hist2_vec[0], hist2_vec.size());

    SyncArray<GHPair> merged_hist(n_bins);
    HistTreeBuilder htb;
    merged_hist.copy_from(htb.merge_historgrams(hists, n_bins));
    auto hist_data = merged_hist.host_data();
    EXPECT_NEAR(hist_data[0].g, 0.21, 1e-5);
    EXPECT_NEAR(hist_data[0].h, 0.42, 1e-5);
    EXPECT_NEAR(hist_data[1].g, 0.63, 1e-5);
    EXPECT_NEAR(hist_data[1].h, 0.84, 1e-5);
    EXPECT_NEAR(hist_data[2].g, 1.05, 1e-5);
    EXPECT_NEAR(hist_data[2].h, 1.26, 1e-5);

//    vector<float> histogram = TreeBuilder().compute_histogram(gradients, splits);
//    EXPECT_EQ(histogram.size(), splits.size() + 1);
//    EXPECT_NEAR(histogram[0], 0.1, 1e-5);
//    EXPECT_NEAR(histogram[1], 0.3, 1e-5);
//    EXPECT_NEAR(histogram[2], 0.4, 1e-5);
}