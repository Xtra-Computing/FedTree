//
// Created by liqinbin on 10/27/20.
//

#ifndef FEDTREE_TREE_BUILDER_H
#define FEDTREE_TREE_BUILDER_H

class TreeBuilder {
    void compute_gain();
    void compute_histogram();
    //support equal division or weighted division
    void propose_split_candidates();
    void update_tree();

};

#endif //FEDTREE_TREE_BUILDER_H
