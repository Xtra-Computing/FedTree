//
// Created by liqinbin on 10/27/20.
//

#ifndef FEDTREE_TREE_BUILDER_H
#define FEDTREE_TREE_BUILDER_H

#include "../common.h"

class TreeBuilder {
    float_type compute_gain(GHPair father, GHPair lch, GHPair rch, float_type lambda);
    void compute_histogram();
    //support equal division or weighted division
    void propose_split_candidates();
    void update_tree();

};

#endif //FEDTREE_TREE_BUILDER_H
