//
// Created by liqinbin on 11/3/20.
//

#ifndef FEDTREE_HIST_TREE_BUILDER_H
#define FEDTREE_HIST_TREE_BUILDER_H

#include "tree_builder.h"
#include "hist_cut.h"

class HistTreeBuilder : public TreeBuilder {
public:

    void init(DataSet &dataset, const GBDTParam &param) override;

    void init(const GBDTParam &param, int n_instances) override;

    void get_bin_ids();

    Tree* build_tree_level_approximate(int level, int round) override;

    void find_split(int level) override;

    void find_split_by_predefined_features(int level) override;

    void compute_histogram_in_a_level(int level, int n_max_splits, int n_bins, int n_nodes_in_level, int* hist_fid_data,
                                      SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist) override;

    void compute_histogram_in_a_node(SyncArray<GHPair> &gradients, HistCut &cut, SyncArray<unsigned char> &dense_bin_id, bool enc);

    void compute_gain_in_a_level(SyncArray<float_type> &gain, int n_nodes_in_level, int n_bins, int* hist_fid_data,
                                 SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist, int n_columns = 0) override;

    void get_best_gain_in_a_level(SyncArray<float_type> &gain, SyncArray<int_float> &best_idx_gain, int n_nodes_in_level, int n_bins) override;

    void get_split_points(SyncArray<int_float> &best_idx_gain, int level, int *hist_fid, SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist);

    SyncArray<GHPair> get_gradients();

    void set_gradients(SyncArray<GHPair> &gh);

    void get_split_points_in_a_node(int node_id, int best_idx, float best_gain, int n_nodes_in_level, int *hist_fid, SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist) override;

    virtual ~HistTreeBuilder() {};

    void update_ins2node_id() override;

    void update_ins2node_id_in_a_node(int node_id) override;

//support equal division or weighted division
    void propose_split_candidates();

    void merge_histograms_server_propose();

    void merge_histograms_client_propose();

    void concat_histograms() override;

    SyncArray<float_type> gain(Tree &tree, SyncArray<GHPair> &hist, int level, int n_split);

    HistCut get_cut() override{
        return cut;
    }

    SyncArray<GHPair> get_hist() override{
        SyncArray<GHPair> h(last_hist.size());
        h.copy_from(last_hist);
        return h;
    }

    HistCut cut;

    void parties_hist_init(int party_size) override{
        LOG(INFO) << "Party size";
        parties_hist.resize(party_size);
        parties_missing_gh.resize(party_size);
        this->party_size = party_size;
        LOG(INFO) << "DOne";
        party_idx = 0;
    }

    void append_hist(SyncArray<GHPair> &hist) override {
        CHECK_LT(party_idx, party_size);
        parties_hist[party_idx].resize(hist.size());
        parties_hist[party_idx].copy_from(hist);
        party_idx += 1;
    }


    void append_hist(SyncArray<GHPair> &hist, SyncArray<GHPair> &missing_gh,int n_partition, int n_max_splits) override{
        CHECK_LT(party_idx, party_size);
        parties_missing_gh[party_idx].resize(n_partition);
        parties_missing_gh[party_idx].copy_from(missing_gh);
        parties_hist[party_idx].resize(n_max_splits);
        parties_hist[party_idx].copy_from(hist);
        party_idx += 1;
    }

    void set_cut (HistCut commonCut) {
        cut.cut_points_val.copy_from(commonCut.cut_points_val);
        cut.cut_col_ptr.copy_from(commonCut.cut_col_ptr);
    }

    void set_last_hist(SyncArray<GHPair> &last_hist_input) {
        last_hist.resize(last_hist_input.size());
        last_hist.copy_from(last_hist_input);
    }

    SyncArray<GHPair> get_last_hist() {
        SyncArray<GHPair> last_hist_return(last_hist.size());
        last_hist_return.copy_from(last_hist);
        return last_hist_return;
    }

    SyncArray<GHPair> get_last_missing_gh() {
        SyncArray<GHPair> last_missing_gh_return(last_missing_gh.size());
        last_missing_gh_return.copy_from(last_missing_gh);
        return last_missing_gh_return;
    }

    void decrypt_histogram(AdditivelyHE::PaillierPrivateKey privateKey) {
        int size = last_hist.size();
        auto hist_data = last_hist.host_data();
        for (int i = 0; i < size; i++) {
            hist_data[i].decrypt(privateKey);
        }
    }

private:
    vector<HistCut> parties_cut;
    // MSyncArray<unsigned char> char_dense_bin_id;
    SyncArray<unsigned char> dense_bin_id;
    SyncArray<GHPair> last_hist;
    SyncArray<GHPair> last_missing_gh;
    MSyncArray<GHPair> parties_hist;
    MSyncArray<GHPair> parties_missing_gh;

    int party_idx = 0;
    int party_size = 0;

    double build_hist_used_time = 0;
    int build_n_hist = 0;
    int total_hist_num = 0;
    double total_dp_time = 0;
    double total_copy_time = 0;

};

#endif //FEDTREE_HIST_TREE_BUILDER_H
