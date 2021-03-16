//
// Created by Tianyuan Fu on 14/3/21.
//

#ifndef FEDTREE_DIFFERENTIALPRIVACY_H
#define FEDTREE_DIFFERENTIALPRIVACY_H

#include <FedTree/syncarray.h>
#include "FedTree/FL/FLparam.h"

//template <typename T>
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

    /**
     * calculates p value based on gain value for each split point
     * @param gain - gain values of all split points in the level
     * @param prob - probability masses (Pi) of all split points in the level (not the actual probability)
     */
    void compute_split_point_probability(SyncArray<float_type> &gain, SyncArray<float_type> &prob);

    /**
     * exponential mechanism: randomly selects split point based on p value
     * @param prob - probability masses (Pi) of all split points in the level (not the actual probability)
     * @param gain - gain values of all split points in the level
     * @param best_idx_gain - mapping from the node index to the gain of split point; containing all the node in the level
     */
    void exponential_select_split_point(SyncArray<float_type> &prob, SyncArray<float_type> &gain,
                                        SyncArray<int_float> &best_idx_gain, int n_nodes_in_level, int n_bins);

    /**
     * add Laplace noise to the data
     * @tparam T - the type of data which shall be added with noise
     * @param data - the data which shall be added noise
     */
    template <typename T>
    void laplace_add_noise(T& data);
};

#endif //FEDTREE_DIFFERENTIALPRIVACY_H
