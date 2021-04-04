//
// Created by Tianyuan Fu on 14/3/21.
//

#include "FedTree/DP/differential_privacy.h"
#include "FedTree/Tree/GBDTparam.h"
#include <numeric>
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
 * @param prob_exponent - exponent for the probability mass; the probability mass is exp(prob_exponent[i])
 */
void DifferentialPrivacy::compute_split_point_probability(SyncArray<float_type> &gain, SyncArray<float_type> &prob_exponent) {

    auto prob_exponent_data = prob_exponent.host_data();
    auto gain_data = gain.host_data();
    for(int i = 0; i < gain.size(); i ++) {
        prob_exponent_data[i] = this->privacy_budget_internal_nodes * fabsf(gain_data[i]) / 2 / delta_g;
//        LOG(INFO) << "budget" << this->privacy_budget_internal_nodes;
//        LOG(INFO) << "gain" << gain_data[i];
    }
}

/**
 * exponential mechanism: randomly selects split point based on p value
 * @param prob_exponent - exponent for the probability mass; the probability mass is exp(prob_exponent[i])
 * @param gain - gain values of all split points in the level
 * @param best_idx_gain - mapping from the node index to the gain of split point; containing all the node in the level
 */
void DifferentialPrivacy::exponential_select_split_point(SyncArray<float_type> &prob_exponent, SyncArray<float_type> &gain,
                                                         SyncArray<int_float> &best_idx_gain, int n_nodes_in_level,
                                                         int n_bins) {

    // initialize randomization
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<> distribution(0.0, 1.0);

    auto prob_exponent_data = prob_exponent.host_data();
    auto gain_data = gain.host_data();
    auto best_idx_gain_data = best_idx_gain.host_data();

    vector<float> probability(n_bins * n_nodes_in_level);

    for (int i = 0; i < n_nodes_in_level; i++) {
        int start = i * n_bins;
        int end = start + n_bins - 1;

        // Given the probability exponent: a, b, c, d
        // The probability[0] can be calculated by exp(a)/(exp(a)+exp(b)+exp(c)+exp(d))
        // To avoid overflow, calculation will be done in 1/(exp(a-a)+exp(b-a)+exp(c-a)+exp(d-a))
        // Probability value with respect to the bin will be stored in probability vector
//        for (int j = start; j <= end; j++) {
//            float curr_exponent = prob_exponent_data[j];
//            float prob_sum_denominator = 0;
//            for (int k = start; k <= end; k++) {
//                prob_sum_denominator += exp(prob_exponent_data[k] - curr_exponent);
//            }
//            probability[j] = 1.0 / prob_sum_denominator;
//        }
        for(int j = start; j <= end; j ++) {
            float curr_exponent = prob_exponent_data[j];
            if(curr_exponent == 0) {
                probability[j] = 0;
            }
            else {
                float denominator = 0;
                for(int k = 0; k <= end; k ++) {
                    if(prob_exponent_data[k] != 0) {
                        denominator += exp(prob_exponent_data[k] - curr_exponent);
                    }
                }
                probability[j] = 1.0 / denominator;
            }
        }

        bool split_selected = false;
        float random_sample = distribution(generator);
        float partial_sum = 0;
        for (int j = start; j <= end; j++) {
            partial_sum += probability[j];
            if (partial_sum > random_sample) {
                best_idx_gain_data[i] = thrust::make_tuple(j, gain_data[j]);
                split_selected = true;
                break;
            }
        }
        if(! split_selected) {
            best_idx_gain_data[i] = thrust::make_tuple(start, 0.0);
        }

//        int max_idx = 0;
//        int max_gain = prob_exponent_data[0];
//        for(int j = start; j <= end; j ++) {
//            if(prob_exponent_data[j] > max_gain) {
//                max_idx = j;
//                max_gain = prob_exponent_data[j];
//            }
//        }
//        best_idx_gain_data[i] = thrust::make_tuple(max_idx, gain_data[max_idx]);
    }
//    for(int i = 0; i < 100; i ++) {
//        LOG(INFO) << "gain" << gain.host_data()[i];
//        LOG(INFO) << "prob" << probability[i];
//    }
    LOG(INFO) << "best rank and gain(dp) "<< best_idx_gain;

}
/**
 * add Laplace noise to the data
 * @param node - the leaf node which noise are to be added
 */
void DifferentialPrivacy::laplace_add_noise(Tree::TreeNode &node) {
    // a Laplace(0, b) variable can be generated by the difference of two i.i.d Exponential(1/b) variables
    float b = this->delta_v/privacy_budget_leaf_nodes;

    std::random_device device;
    std::mt19937 generator(device());
    std::exponential_distribution<double> distribution(1.0/b);

    double noise = distribution(generator) - distribution(generator);
    node.base_weight += noise;
}
