
//
// Created by liqinbin on 10/27/20.
//

#ifndef FEDTREE_TREE_BUILDER_H
#define FEDTREE_TREE_BUILDER_H

#include "FedTree/dataset.h"
#include "FedTree/Encryption/HE.h"
#include "function_builder.h"
#include "tree.h"
#include "splitpoint.h"
#include "hist_cut.h"

class TreeBuilder : public FunctionBuilder{
public:
    virtual void find_split(int level) = 0;

    virtual void find_split_by_predefined_features(int level) = 0;

    virtual void update_ins2node_id() = 0;

    vector<Tree> build_approximate(const SyncArray<GHPair> &gradients, bool update_y_predict = true) override;

    void build_tree_by_predefined_structure(const SyncArray<GHPair> &gradients, vector<Tree> &trees);

    void build_init(const SyncArray<GHPair> &gradients, int k) override;

    void init(DataSet &dataset, const GBDTParam &param) override;

    virtual void update_tree();

    virtual void update_tree_by_sp_values();

    void predict_in_training(int k);

//    virtual void split_point_all_reduce(int depth);
    // Refer to ThunderGBM hist_tree_builder.cu find_split

//    void get_split(int level, int device_id);

    void find_split (SyncArray<SplitPoint> &sp, int n_nodes_in_level, Tree tree, SyncArray<int_float> best_idx_gain, int nid_offset, HistCut cut, SyncArray<GHPair> hist, int n_bins);



    void merge_histograms();

    void update_gradients(SyncArray<GHPair> &gradients, SyncArray<float_type> &y, SyncArray<float_type> &y_p);


//    virtual void init(const DataSet &dataset, const GBDTParam &param) {
//        this->param = param;
//    };


//for multi-device
//    virtual void ins2node_id_all_reduce(int depth);



//    virtual void split_point_all_reduce(int depth);

    virtual ~TreeBuilder(){};

    SyncArray<GHPair> gradients;

    Tree get_trees() {
        return trees;
    }

    void update_trees(Tree trees_updated) {
        trees = trees_updated;
    }

protected:
    int n_instances;
    Tree trees;
    SyncArray<int> ins2node_id;
    SyncArray<SplitPoint> sp;
    bool has_split;
//    vector<Shard> shards;
//    DataSet* dataset;
    DataSet sorted_dataset;
};

#endif //FEDTREE_TREE_BUILDER_H