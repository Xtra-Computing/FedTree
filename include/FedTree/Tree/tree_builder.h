
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

    virtual void update_ins2node_id() = 0;

    vector<Tree> build_approximate(const SyncArray<GHPair> &gradients) override;

    void init(DataSet &dataset, const GBDTParam &param) override;

    virtual void update_tree();

    void predict_in_training(int k);

    virtual void split_point_all_reduce(int depth);
    // Refer to ThunderGBM hist_tree_builder.cu find_split

    void encrypt_gradients(AdditivelyHE::PaillierPublicKey pk);

    SyncArray<GHPair> get_gradients();

    void set_gradients(SyncArray<GHPair> &gh);

    void get_split(int level, int device_id);

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

protected:
    int n_instances;
    Tree trees;
    SyncArray<int> ins2node_id;
    SyncArray<SplitPoint> sp;
    SyncArray<GHPair> gradients;
    bool has_split;
//    vector<Shard> shards;
    DataSet* dataset;
};

#endif //FEDTREE_TREE_BUILDER_H