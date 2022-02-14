//
// Created by Tianyuan Fu on 14/3/21.
//

#ifndef FEDTREE_DIFFERENTIALPRIVACY_H
#define FEDTREE_DIFFERENTIALPRIVACY_H

#include <FedTree/syncarray.h>
#include "FedTree/FL/FLparam.h"
#include "FedTree/Tree/tree.h"
#include <algorithm>

using namespace std;
//template <typename T>
class DifferentialPrivacy {
public:
    float max_gradient = 1.0;
    float lambda;
    float constant_h = 1.0;
    float delta_g;
    float delta_v;
    float privacy_budget;
    float privacy_budget_per_tree;
    float privacy_budget_leaf_nodes;
    float privacy_budget_internal_nodes;

    void init(FLParam fLparam);

    /**
     * calculates p value based on gain value for each split point
     * @param gain - gain values of all split points in the level
     * @param prob - probability masses (Pi) of all split points in the level (not the actual probability)
     */
    void compute_split_point_probability(SyncArray<float_type> &gain, SyncArray<float_type> &prob_exponent);

    /**
     * exponential mechanism: randomly selects split point based on p value
     * @param prob - probability masses (Pi) of all split points in the level (not the actual probability)
     * @param gain - gain values of all split points in the level
     * @param best_idx_gain - mapping from the node index to the gain of split point; containing all the node in the level
     */
    void exponential_select_split_point(SyncArray<float_type> &prob_exponent, SyncArray<float_type> &gain,
                                        SyncArray<int_float> &best_idx_gain, int n_nodes_in_level, int n_bins);

    /**
     * adds Laplace noise to the data
     * @param node - the leaf node which noise are to be added
     */
    void laplace_add_noise(Tree::TreeNode &node);

    /**
     * clips gradient data
     * @param value - gradient data
     */
    template <typename T>
    void clip_gradient_value(T& value) {
        value = max<T>(min<T>(value, 1),-1);
    }
};

#endif //FEDTREE_DIFFERENTIALPRIVACY_H
