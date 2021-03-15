//
// Created by Tianyuan Fu on 14/3/21.
//

#include "FedTree/DP/DifferentialPrivacy.h"
#include "FedTree/Tree/GBDTparam.h"
#include <random>
#include <math.h>

void DifferentialPrivacy::init(FLParam flparam) {
    GBDTParam gbdt_param = flparam.gbdt_param;
    this->lambda = gbdt_param.lambda;
    this->delta_g = 3 * this->max_gradient * this->max_gradient;
    this->delta_v = this->max_gradient / (1.0+this->lambda);
    this->privacy_budget = flparam.privacy_budget;
    this->privacy_budget_per_tree = this->privacy_budget / gbdt_param.n_trees;
    this->privacy_budget_leaf_nodes = this->privacy_budget_per_tree / 2.0;
    this->privacy_budget_internal_nodes = this->privacy_budget_per_tree / 2.0 / gbdt_param.depth;
}

/**
 * calculates p value based on gain value for each split point
 * @param gain - gain values of all split points in the level
 * @param prob - probability masses (Pi) of all split points in the level (not the actual probability)
 */
void DifferentialPrivacy::compute_split_point_probability(SyncArray<float_type> &gain, SyncArray<float_type> &prob) {
    auto prob_data = prob.host_data();
    auto gain_data = gain.host_data();
    for(int i = 0; i < gain.size(); i ++) {
        prob_data[i] = exp(this->privacy_budget_leaf_nodes * gain_data[i] / 2 / delta_g);
    }
}

/**
 * exponential mechanism: randomly selects split point based on p value
 * @param prob - probability masses (Pi) of all split points in the level (not the actual probability)
 * @param gain - gain values of all split points in the level
 * @param best_idx_gain - mapping from the node index to the gain of split point; containing all the node in the level
 */
void DifferentialPrivacy::exponential_select_split_point(SyncArray<float_type> &prob, SyncArray<float_type> &gain,
                                                          SyncArray<int_float> &best_idx_gain, int n_nodes_in_level,
                                                          int n_bins) {
    std::random_device device;
    std::mt19937 generator(device);
    std::uniform_real_distribution<> distribution(0.0, 1.0);

    auto prob_data = prob.host_data();
    auto gain_data = gain.host_data();
    auto best_idx_gain_data = best_idx_gain.host_data();

    for(int i = 0; i < n_nodes_in_level; i ++) {
        int start = i * n_bins;
        int end = start + n_bins - 1;

        float prob_sum = std::accumulate(prob_data[start], prob_data[end], 0);

        float rand_sample = distribution(generator);

        float partial_sum = 0;
        for(int j = start; j <= end; j ++) {
            partial_sum += prob_data[j];
            if(partial_sum/prob_sum > rand_sample) {
                best_idx_gain_data[i] = thrust::make_tuple(j, gain_data[j]);
                break;
            }
        }
    }
}