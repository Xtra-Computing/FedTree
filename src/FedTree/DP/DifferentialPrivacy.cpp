//
// Created by Tianyuan Fu on 14/3/21.
//

#include "FedTree/DP/DifferentialPrivacy.h"
#include "FedTree/Tree/GBDTparam.h"
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

void DifferentialPrivacy::compute_split_point_probability(SyncArray<float_type> &gain, SyncArray<float_type> &prob) {
    auto prob_data = prob.host_data();
    auto gain_data = gain.host_data();
    float prob_sum = 0.0;
    for(int i = 0; i < gain.size(); i ++) {
        prob_data[i] = exp(this->privacy_budget_leaf_nodes * gain_data[i] / 2 / delta_g);
        prob_sum += prob_data[i];
    }
    for(int i = 0; i < prob.size(); i ++) {
        prob_data[i] /= prob_sum;
    }
}