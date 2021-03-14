//
// Created by Tianyuan Fu on 14/3/21.
//

#ifndef FEDTREE_DIFFERENTIALPRIVACY_H
#define FEDTREE_DIFFERENTIALPRIVACY_H

#include "FedTree/FL/FLparam.h"

class DifferentialPrivacy {
public:
    float max_gradient = 1.0;
    float lambda;
    float delta_g;
    float delta_v;
    float privacy_budget;
    float privacy_budget_per_tree;
    float privacy_budget_leaf_nodes;
    float privacy_budget_internal_nodes;

    void init(FLParam fLparam);
};

#endif //FEDTREE_DIFFERENTIALPRIVACY_H
