
//
// Created by liqinbin on 10/27/20.
//

#ifndef FEDTREE_TREE_BUILDER_H
#define FEDTREE_TREE_BUILDER_H

#include "FedTree/dataset.h"
#include "tree.h"
#include "splitpoint.h"
#include "hist_cut.h"

class TreeBuilder : public FuncionBuilder{
public:
    virtual void find_split(int level) = 0;

    virtual void update_ins2node_id() = 0;

    vector<Tree> build_approximate(const SyncArray<GHPair> &gradients) override;

    void init(const DataSet &dataset, const GBMParam &param) override;

    // Refer to ThunderGBM hist_tree_builder.cu find_split


    float_type compute_gain(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight, float_type lambda);

//    SyncArray<float_type> gain(Tree &tree, SyncArray<GHPair> &hist, int level, int n_split);

    void get_split(int level, int device_id);

    SyncArray<int_float> best_idx_gain(SyncArray<float_type> &gain, int n_bins, int level, int n_split);

    void find_split (SyncArray<SplitPoint> &sp, int n_nodes_in_level, Tree tree, SyncArray<int_float> best_idx_gain, int nid_offset, HistCut cut, SyncArray<GHPair> hist, int n_bins);

    void update_tree(SyncArray<SplitPoint> sp, Tree &tree, float_type lambda, float_type rt_eps);

    void merge_histograms();

    void update_gradients(SyncArray<GHPair> &gradients, SyncArray<float_type> &y, SyncArray<float_type> &y_p);

//    virtual void update_ins2node_id() = 0;

//    virtual void init(const DataSet &dataset, const GBDTParam &param) {
//        this->param = param;
//    };

    virtual const SyncArray<float_type> &get_y_predict(){ return y_predict; };

    static TreeBuilder *create(std::string name);

//    virtual void update_tree();

//    void predict_in_training(int k);

//    virtual void split_point_all_reduce(int depth);

//    virtual void ins2node_id_all_reduce(int depth);

    virtual ~TreeBuilder(){};

protected:
    SyncArray<float_type> y_predict;
    SyncArray<GHPair> gradients;
    GBDTParam param;
//    vector<Shard> shards;
    DataSet& dataset;
    int n_instances;
    Tree trees;

    SyncArray<int> ins2node_id;
    SyncArray<SplitPoint> sp;
//    SyncArray<GHPair> gradients;
    bool has_split;
};

#endif //FEDTREE_TREE_BUILDER_H