//
// Created by hanyuxuan on 28/10/20.
//

#include "gtest/gtest.h"
#include "FedTree/Tree/tree_builder.h"
#include "FedTree/Tree/hist_tree_builder.h"
#include "FedTree/Tree/hist_cut.h"
#include "FedTree/Tree/GBDTparam.h"
#include <string>

class TreeBuilderTest : public ::testing::Test {
public:

    GHPair father;
    GHPair lch;
    GHPair rch;
    HistTreeBuilder treeBuilder;

protected:
    void SetUp() override {
        father = GHPair(5.1, 5);
        lch = GHPair(-2.1, 2);
        rch = GHPair(9.61, 3);
    }
};

//TEST_F(TreeBuilderTest, compute_gain) {
//    EXPECT_FLOAT_EQ(treeBuilder.compute_gain_in_a_node(father, lch, rch, -5, 0.1), 26.791);
//}

TEST_F(TreeBuilderTest, gain_per_level) {
    SyncArray<GHPair> gradients(4);
    const vector<GHPair> gradients_vec = {GHPair(0.1, 0.2), GHPair(0.3, 0.4), GHPair(0.5, 0.6), GHPair(0.7, 0.8)};
    gradients.copy_from(&gradients_vec[0], gradients_vec.size());
    HistTreeBuilder htb;
    Tree tree;
    GBDTParam param = GBDTParam();
    param.depth = 2;
    param.min_child_weight = 0.0;
    param.lambda = 0.2;
    tree.init_CPU(gradients, param);
    SyncArray<GHPair> hist(2);
    const vector<GHPair> hist_vec = {GHPair(0.2, 0.2), GHPair(0.5, 0.5)};
    hist.copy_from(&hist_vec[0], hist_vec.size());
    auto result = htb.gain(tree, hist, 0, 2);
    EXPECT_EQ(result.size(), 2);
    EXPECT_FLOAT_EQ(result.host_data()[0], 0);
    EXPECT_FLOAT_EQ(result.host_data()[1], 0);
}

TEST_F(TreeBuilderTest, compute_histogram) {
    printf("### Test compute_histogram ###\n");
    int n_instances = 4;
    int n_columns = 2;

    SyncArray<GHPair> gradients(4);
    const vector<GHPair> gradients_vec = {GHPair(0.1, 0.2), GHPair(0.3, 0.4), GHPair(0.5, 0.6), GHPair(0.7, 0.8)};
    gradients.copy_from(&gradients_vec[0], gradients_vec.size());

    HistCut cut;
    cut.cut_col_ptr = SyncArray<int>(3);
    const vector<int> cut_col_ptr_vec = {0, 1, 3};
    cut.cut_col_ptr.copy_from(&cut_col_ptr_vec[0], cut_col_ptr_vec.size());

    SyncArray<unsigned char> dense_bin_id(8);
    const vector<unsigned char> bin_vec = {0, 0, 0, 1, 1, 1, 1, 2};
    dense_bin_id.copy_from(&bin_vec[0], bin_vec.size());

    SyncArray<GHPair> hist(5);
    HistTreeBuilder htb;
    htb.compute_histogram_in_a_node(gradients, cut, dense_bin_id);
    hist.copy_from(htb.get_hist());
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

TEST_F(TreeBuilderTest, merge_histogram_server) {
    printf("### Test merge_histogram ###\n");

    const vector<GHPair> hist1_vec = {GHPair(0.1, 0.2), GHPair(0.3, 0.4), GHPair(0.5, 0.6)};
    const vector<GHPair> hist2_vec = {GHPair(0.11, 0.22), GHPair(0.33, 0.44), GHPair(0.55, 0.66)};

    MSyncArray<GHPair> hists(2, 3);
    hists[0].copy_from(&hist1_vec[0], hist1_vec.size());
    hists[1].copy_from(&hist2_vec[0], hist2_vec.size());

    SyncArray<GHPair> merged_hist(3);
    HistTreeBuilder htb;
    htb.parties_hist_init(2);
    htb.append_hist(hists[0]);
    htb.append_hist(hists[1]);
    htb.merge_histograms_server_propose(merged_hist, merged_hist);
    merged_hist.copy_from(htb.get_hist());
    auto hist_data = merged_hist.host_data();
    EXPECT_NEAR(hist_data[0].g, 0.21, 1e-5);
    EXPECT_NEAR(hist_data[0].h, 0.42, 1e-5);
    EXPECT_NEAR(hist_data[1].g, 0.63, 1e-5);
    EXPECT_NEAR(hist_data[1].h, 0.84, 1e-5);
    EXPECT_NEAR(hist_data[2].g, 1.05, 1e-5);
    EXPECT_NEAR(hist_data[2].h, 1.26, 1e-5);
}

/*
TEST_F(TreeBuilderTest, merge_histogram_clients) {
    printf("### Test merge_histogram clients###\n");

    vector<GHPair> hist1_vec;
    vector<GHPair> hist2_vec;
    for (int i = 0; i < 14; i++) {
        hist1_vec.push_back(GHPair(1, 1));
        hist2_vec.push_back(GHPair(1, 1));
    }

    MSyncArray<GHPair> hists(2, 14);
    hists[0].copy_from(&hist1_vec[0], hist1_vec.size());
    hists[1].copy_from(&hist2_vec[0], hist2_vec.size());


    const vector<float_type> cut_points_val_vec1 = {0.1, 0.3, 5, 7, 9, 15, 25, 35, 10, 11};
    const vector<int> cut_ptr_vec1 = {0, 2, 5, 8, 10};
    const vector<float_type> cut_points_val_vec2 = {0.4, 0.5, 0.6, 4, 8, 30, 50, 9, 12, 15};
    const vector<int> cut_ptr_vec2 = {0, 3, 5, 7, 10};

    vector<HistCut> cuts(2);
    cuts[0].cut_col_ptr = SyncArray<int>(5);
    cuts[0].cut_col_ptr.copy_from(&cut_ptr_vec1[0], cut_ptr_vec1.size());
    cuts[0].cut_points_val = SyncArray<float_type>(10);
    cuts[0].cut_points_val.copy_from(&cut_points_val_vec1[0], cut_points_val_vec1.size());
    cuts[1].cut_col_ptr = SyncArray<int>(5);
    cuts[1].cut_col_ptr.copy_from(&cut_ptr_vec2[0], cut_ptr_vec2.size());
    cuts[1].cut_points_val = SyncArray<float_type>(10);
    cuts[1].cut_points_val.copy_from(&cut_points_val_vec2[0], cut_points_val_vec2.size());

//    HistTreeBuilder htb;
//    EXPECT_FLOAT_EQ(htb.merge_histograms_client_propose(hists, cuts)[0], -0.1);
//    EXPECT_FLOAT_EQ(htb.merge_histograms_client_propose(hists, cuts)[7], 0.6);
//    EXPECT_FLOAT_EQ(htb.merge_histograms_client_propose(hists, cuts)[8], 0);
//    EXPECT_FLOAT_EQ(htb.merge_histograms_client_propose(hists, cuts)[9], 2);

    SyncArray<GHPair> merged_hist(31);
    HistTreeBuilder htb;
    htb.merge_histograms_client_propose(hists, cuts, false);
    merged_hist.copy_from(htb.get_hist());
    auto hist_data = merged_hist.host_data();
    EXPECT_NEAR(hist_data[0].g, 0.5, 1e-5);
    EXPECT_NEAR(hist_data[1].g, 0.5, 1e-5);
    EXPECT_NEAR(hist_data[2].g, 0.5, 1e-5);
    EXPECT_NEAR(hist_data[3].g, 0.5, 1e-5);
    EXPECT_NEAR(hist_data[4].g, 1.5, 1e-5);
    EXPECT_NEAR(hist_data[5].g, 1.5, 1e-5);
    EXPECT_NEAR(hist_data[6].g, 1, 1e-5);
}
*/
