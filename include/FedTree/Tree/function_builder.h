//
// Created by liqinbin on 12/11/20.
//

#ifndef FEDTREE_FUNCTION_BUILDER_H
#define FEDTREE_FUNCTION_BUILDER_H


#include "tree.h"
#include "FedTree/common.h"
#include "FedTree/dataset.h"
//#include "FedTree/Encryption/HE.h"
#include "FedTree/Tree/hist_cut.h"

class FunctionBuilder {
public:
    virtual vector<Tree> build_approximate(const SyncArray<GHPair> &gradients, bool update_y_predict = true) = 0;

    virtual Tree* build_tree_level_approximate(int level, int round) = 0;

    virtual Tree get_tree()= 0;

    virtual void set_tree(Tree tree) = 0;

    virtual void set_y_predict(int k) = 0;

    virtual void build_init(const SyncArray<GHPair> &gradients, int k) = 0;

    virtual void compute_histogram_in_a_level(int level, int n_max_splits, int n_bins, int n_nodes_in_level,
                                              int *hist_fid_data, SyncArray<GHPair> &missing_gh,
                                              SyncArray<GHPair> &hist) = 0;

    virtual void compute_gain_in_a_level(SyncArray<float_type> &gain, int n_nodes_in_level, int n_bins,
                                         int *hist_fid_data, SyncArray<GHPair> &missing_gh,
                                         SyncArray<GHPair> &hist, int n_column) = 0;

    virtual void get_best_gain_in_a_level(SyncArray<float_type> &gain, SyncArray<int_float> &best_idx_gain,
                                          int n_nodes_in_level, int n_bins) = 0;

    virtual void get_split_points_in_a_node(int node_id, int best_idx, float best_gain, int n_nodes_in_level,
                                            int *hist_fid, SyncArray<GHPair> &missing_gh,
                                            SyncArray<GHPair> &hist) = 0;

    virtual void update_ins2node_id_in_a_node(int node_id) = 0;

    virtual HistCut get_cut() = 0;

    virtual SyncArray<GHPair> get_hist() = 0;

    virtual void parties_hist_init(int party_size) = 0;

    virtual void append_hist(SyncArray<GHPair> &hist) = 0;

    virtual void concat_histograms() = 0;

    virtual void init(DataSet &dataset, const GBDTParam &param) {
        this->param = param;
    };

    virtual const SyncArray<float_type> &get_y_predict() { return y_predict; };

    virtual ~FunctionBuilder() {};

    static FunctionBuilder *create(std::string name);

protected:
    SyncArray<float_type> y_predict;
    GBDTParam param;
};


#endif //FEDTREE_FUNCTION_BUILDER_H
