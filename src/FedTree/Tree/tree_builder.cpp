//
// Created by liqinbin on 11/3/20.
//

#include "FedTree/Tree/tree_builder.h"
#include "FedTree/Tree/hist_tree_builder.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/discard_iterator.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <math.h>

float_type TreeBuilder:: compute_gain(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight, float_type lambda) {
    if (lch.h >= min_child_weight && rch.h >= min_child_weight) {
        return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda) -
               (father.g * father.g) / (father.h + lambda);
    } else {
        return 0;
    }
}

int TreeBuilder:: get_nid(int index) {
    return 0;
}

int TreeBuilder:: get_pid(int index) {
    return 0;
}

SyncArray<float_type> TreeBuilder::gain(Tree tree, int n_split, int n_partition, int n_max_splits) {
    SyncArray<float_type> gain(n_max_splits);
    const Tree::TreeNode *nodes_data = tree.nodes.device_data();
    float_type mcw = this->param.min_child_weight;
    float_type l = this->param.lambda;
    SyncArray<GHPair> missing_gh(n_partition);
    const auto missing_gh_data = missing_gh.device_data();
    SyncArray<GHPair> hist(n_max_splits);
    GHPair *gh_prefix_sum_data = hist.device_data();
    float_type *gain_data = gain.device_data();
    for (int i = 0; i < n_split; i++) {
        int nid = get_nid(i);
        int pid = get_pid(i);
        if (nodes_data[nid].is_valid) {
            GHPair father_gh = nodes_data[nid].sum_gh_pair;
            GHPair p_missing_gh = missing_gh_data[pid];
            GHPair rch_gh = gh_prefix_sum_data[i];
            float_type default_to_left_gain = std::max(0.f,
                                                  compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
            rch_gh = rch_gh + p_missing_gh;
            float_type default_to_right_gain = std::max(0.f,
                                                   compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
            if (default_to_left_gain > default_to_right_gain) {
                gain_data[i] = default_to_left_gain;
            } else {
                gain_data[i] = -default_to_right_gain;//negative means default split to right
            }
        } else {
            gain_data[i] = 0;
        }
    }
    return gain;
}


SyncArray<int_float> TreeBuilder::best_idx_gain(SyncArray<float_type> gain, int n_nodes_in_level, int n_split) {
    SyncArray<int_float> best_idx_gain(n_nodes_in_level);
    auto nid = [this](int index) {
        return get_nid(index);
    };
    auto arg_abs_max = [](const int_float &a, const int_float &b) {
        if (fabsf(thrust::get<1>(a)) == fabsf(thrust::get<1>(b)))
            return thrust::get<0>(a) < thrust::get<0>(b) ? a : b;
        else
            return fabsf(thrust::get<1>(a)) > fabsf(thrust::get<1>(b)) ? a : b;
    };
    auto nid_iterator = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), nid);
    reduce_by_key(
            thrust::host,
            nid_iterator, nid_iterator + n_split,
            make_zip_iterator(make_tuple(thrust::counting_iterator<int>(0), gain.device_data())),
            thrust::make_discard_iterator(),
            best_idx_gain.device_data(),
            thrust::equal_to<int>(),
            arg_abs_max
    );

    return best_idx_gain;
}


TreeBuilder *TreeBuilder::create(std::string name) {
    if (name == "hist") return new HistTreeBuilder;
    LOG(FATAL) << "unknown builder " << name;
    return nullptr;
}

