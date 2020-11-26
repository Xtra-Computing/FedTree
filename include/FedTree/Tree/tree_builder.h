
//
// Created by liqinbin on 10/27/20.
//

#ifndef FEDTREE_TREE_BUILDER_H
#define FEDTREE_TREE_BUILDER_H

#include "FedTree/dataset.h"
#include "tree.h"

class TreeBuilder {
public:
    // Refer to ThunderGBM hist_tree_builder.cu find_split
    void compute_histogram();

    float_type compute_gain(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight, float_type lambda);

    SyncArray<float_type> gain(Tree tree, SyncArray<GHPair> hist, int level, int n_split);

    int get_nid(int index, int level);

    int get_pid(int index);

    void get_split(int level, int device_id);

    SyncArray<int_float> best_idx_gain(SyncArray<float_type> gain, int level, int n_nodes_in_level, int n_split);

    void update_tree();


    void merge_histograms();


//    virtual void find_split(int level, int device_id) = 0;

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
    GBDTParam param;
//    vector<Shard> shards;
//    int n_instances;
    vector<Tree> trees;
//    SyncArray<int> ins2node_id;
//    SyncArray<SplitPoint> sp;
//    SyncArray<GHPair> gradients;
//    vector<bool> has_split;
};

#endif //FEDTREE_TREE_BUILDER_H