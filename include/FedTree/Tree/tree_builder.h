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
    void compute_gain();
    void get_split();
    void update_tree();

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
//    vector<Tree> trees;
//    SyncArray<int> ins2node_id;
//    SyncArray<SplitPoint> sp;
//    SyncArray<GHPair> gradients;
//    vector<bool> has_split;
};

#endif //FEDTREE_TREE_BUILDER_H
