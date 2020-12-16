//
// Created by liqinbin on 11/3/20.
//

#ifndef FEDTREE_HIST_TREE_BUILDER_H
#define FEDTREE_HIST_TREE_BUILDER_H

#include "tree_builder.h"
#include "hist_cut.h"

class HistTreeBuilder : public TreeBuilder {
public:


    void init(const DataSet &dataset, const GBDTParam &param) override;

    void get_bin_ids();

    void find_split(int level) override;

    void compute_histogram_in_a_level(int level, int n_max_splits, int n_bins, int n_nodes_in_level, transform_iterator& hist_fid);

    void compute_histogram_in_a_node(SyncArray<GHPair> &gradients, HistCut &cut, SyncArray<unsigned char> &dense_bin_id, bool enc);

    void compute_gain_in_a_level(SyncArrary<float_type> &gain, int n_max_splits, int n_bins, transform_iterator& hist_fid);

    void get_best_gain_in_a_level(SyncArray<float_type> &gain, SyncArray<int_float> &best_idx_gain, int n_nodes_in_level, int n_bins);

    void get_split_points(SyncArray<int_float> &best_idx_gain);

    SyncArray<GHPair> compute_histogram(SyncArray<GHPair> &gradients, HistCut &cut,
                                        SyncArray<unsigned char> &dense_bin_id);


    virtual ~HistTreeBuilder() {};

    void update_ins2node_id() override;

//support equal division or weighted division
    void propose_split_candidates();

    void merge_histograms_server_propose(MSyncArray<GHPair> &histograms, bool enc);

    void merge_histograms_client_propose(MSyncArray<GHPair> &histograms, vector<HistCut> &cuts, bool enc);

    SyncArray<GHPair> get_hist() {
        SyncArray<GHPair> h(last_hist.size());
        h.copy_from(last_hist);
        return h;
    }

private:
    HistCut cut;
    // MSyncArray<unsigned char> char_dense_bin_id;
    SyncArray<unsigned char> dense_bin_id;
    SyncArray<GHPair> last_hist;

    double build_hist_used_time = 0;
    int build_n_hist = 0;
    int total_hist_num = 0;
    double total_dp_time = 0;
    double total_copy_time = 0;
};

#endif //FEDTREE_HIST_TREE_BUILDER_H
