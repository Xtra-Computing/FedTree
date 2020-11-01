//
// Created by liqinbin on 10/27/20.
//

#ifndef FEDTREE_TREE_BUILDER_H
#define FEDTREE_TREE_BUILDER_H

#include "FedTree/common.h"

class TreeBuilder {
public:
    void compute_gain();
    vector<float> compute_histogram(vector<float> gradients, vector<int> splits);
    //support equal division or weighted division
    void propose_split_candidates();
    void update_tree();
};

#endif //FEDTREE_TREE_BUILDER_H
