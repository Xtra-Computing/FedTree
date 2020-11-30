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

float_type
TreeBuilder::compute_gain(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight, float_type lambda) {
    if (lch.h >= min_child_weight && rch.h >= min_child_weight) {
        return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda) -
               (father.g * father.g) / (father.h + lambda);
    } else {
        return 0;
    }
}

int TreeBuilder::get_nid(int index) {
    return 0;
}

int TreeBuilder::get_pid(int index) {
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

/*
// need to know the split format in order to proceed
void TreeBuilder::update_tree(SyncArray<float> &gain, SyncArray<> &split, Tree &tree, float rt_eps, float lambda) {
    int n_nodes_in_level = split.size();
    Tree::TreeNode *nodes_data = tree.nodes;
    for (int i = 0; i < n_nodes_in_level; i++) {
        float best_split_gain = gain[i];
        if (best_split_gain > rt_eps) {
            if (split_data[i].nid == -1) return;
            int nid = split_data[i].nid;
            Tree::TreeNode &node = nodes_data[nid];
            node.gain = best_split_gain;
            // left children
            Tree::TreeNode &lch = nodes_data[node.lch_index];
            // right children
            Tree::TreeNode &rch = nodes_data[node.rch_index];
            lch.is_valid = true;
            rch.is_valid = true;
            node.split_feature_id = split_data[i].split_fea_id;
            // Gradient Hessian Pair
            GHPair p_missing_gh = split_data[i].fea_missing_gh;
            node.split_value = split_data[i].fval;
            node.split_bid = split_data[i].split_bid;
            rch.sum_gh_pair = split_data[i].rch_sum_gh;
            if (split_data[i].default_right) {
                rch.sum_gh_pair = rch.sum_gh_pair + p_missing_gh;
                node.default_right = true;
            }
            lch.sum_gh_pair = node.sum_gh_pair - rch.sum_gh_pair;
            lch.calc_weight(lambda);
            rch.calc_weight(lambda);
        }else {
            //set leaf
            if (split_data[i].nid == -1) return;
            int nid = split_data[i].nid;
            Tree::TreeNode &node = nodes_data[nid];
            node.is_leaf = true;
            nodes_data[node.lch_index].is_valid = false;
            nodes_data[node.rch_index].is_valid = false;
        }
    }
}
*/

SyncArray<GHPair>
HistTreeBuilder::compute_histogram(SyncArray<GHPair> &gradients, HistCut &cut,
                                   SyncArray<unsigned char> &dense_bin_id) {
    int n_columns = cut.cut_row_ptr.size() - 1;
    int n_instances = dense_bin_id.size() / n_columns;
    auto gh_data = gradients.host_data();
    auto cut_row_ptr_data = cut.cut_row_ptr.host_data();
    auto dense_bin_id_data = dense_bin_id.host_data();
    int n_bins = n_columns + cut_row_ptr_data[n_columns];

    SyncArray<GHPair> hist(n_bins);
    auto hist_data = hist.host_data();

    for (int i = 0; i < n_instances * n_columns; i++) {
        int iid = i / n_columns;
        int fid = i % n_columns;
        unsigned char bid = dense_bin_id_data[iid * n_columns + fid];

        int feature_offset = cut_row_ptr_data[fid] + fid;
        const GHPair src = gh_data[iid];
        GHPair &dest = hist_data[feature_offset + bid];
        if (src.h != 0)
            dest.h += src.h;
        if (src.g != 0)
            dest.g += src.g;
    }

    return hist;
}

//assumption: GHPairs in the histograms of all clients are arranged in the same order

SyncArray<GHPair>
HistTreeBuilder::merge_histograms_server_propose(MSyncArray<GHPair> &histograms) {

    int n_bins = histograms[0].size();
    SyncArray<GHPair> merged_hist(n_bins);
    auto merged_hist_data = merged_hist.host_data();

    for (int i = 0; i < histograms.size(); i++) {
        auto hist_data = histograms[i].host_data();
        for (int j = 0; j < n_bins; j++) {
            GHPair &src = hist_data[j];
            GHPair &dest = merged_hist_data[j];
            if (src.h != 0)
                dest.h += src.h;
            if (src.g != 0)
                dest.g += src.g;
        }
    }

    return merged_hist;
}


//assumption 1: bin sizes for the split of a feature are the same
//assumption 2: for each feature, there must be at least 3 bins (2 cut points)
//assumption 3: cut_val_data is sorted by feature id and split value, eg: [f0(0.1), f0(0.2), f0(0.3), f1(100), f1(200),...]
//assumption 4: gradients and hessians are near uniformly distributed

SyncArray<GHPair>
HistTreeBuilder::merge_histograms_client_propose(MSyncArray<GHPair> &histograms, vector<HistCut> &cuts) {
    CHECK_EQ(histograms.size(), cuts.size());
    int n_columns = cuts[0].cut_row_ptr.size() - 1;
    vector<float_type> low(n_columns, std::numeric_limits<float>::max());
    vector<float_type> high(n_columns, -std::numeric_limits<float>::max());
    vector<float_type> resolution(n_columns, std::numeric_limits<float>::max());
    vector<vector<float_type>> bin_edges;
    for (int i = 0; i < cuts.size(); i++) {
        auto cut_val_data = cuts[i].cut_points_val.host_data();
        auto cut_row_ptr_data = cuts[i].cut_row_ptr.host_data();
        vector<float_type> v = {};
        for (int j = 0; j < n_columns; j++) {
            int end = cut_row_ptr_data[j + 1];
            int start = cut_row_ptr_data[j];
            float_type res = cut_val_data[end - 1] - cut_val_data[end - 2];
            resolution[j] = std::min(res, resolution[j]);
            float_type l = cut_val_data[start] - res;
            low[j] = std::min(l, low[j]);
            float_type h = cut_val_data[end - 1] + res;
            high[j] = std::max(h, high[j]);
            for (int k = -1; k < end - start + 1; k++)
                v.push_back(cut_val_data[start] + k * res);
        }
        bin_edges.push_back(v);
    }

//    return bin_edges;
    int n_bins = 0;
    vector<float_type> merged_bin_edges;
    vector<int> merged_bins_count;
    merged_bins_count.push_back(0);
    for (int i = 0; i < n_columns; i++) {
        float_type count = (high[i] - low[i]) / resolution[i];
        if (abs(int(count) - count) < 1e-6)
            count = int(count);
        else
            count = ceil(count);
        n_bins += count;
        merged_bins_count.push_back(n_bins);
        for (int j = 0; j <= count; j++)
            merged_bin_edges.push_back(std::min(low[i] + j * resolution[i], high[i]));
    }
//    return merged_bin_edges;

    SyncArray<GHPair> merged_hist(n_bins);
    auto merged_hist_data = merged_hist.host_data();
    for (int i = 0; i < histograms.size(); i++) {
        CHECK_EQ(histograms[i].size(), cuts[i].cut_points_val.size() + n_columns);
        CHECK_EQ(histograms[i].size() + n_columns, bin_edges[i].size());
        auto hist_data = histograms[i].host_data();
        auto cut_row_ptr_data = cuts[i].cut_row_ptr.host_data();
        for (int j = 0; j < n_columns; j++) {
            int client_bin_index_low = cut_row_ptr_data[j] + 2 * j;
            int client_bin_index_high = cut_row_ptr_data[j + 1] + 2 * (j + 1);
            for (int k = merged_bins_count[j]; k < merged_bins_count[j + 1]; k++) {
                float_type bin_low = merged_bin_edges[k + j];
                float_type bin_high = merged_bin_edges[k + j + 1];
                for (int m = client_bin_index_low; m < client_bin_index_high - 1; m++) {
                    float_type client_low = bin_edges[i][m];
                    float_type client_high = bin_edges[i][m + 1];
                    if (bin_low < client_low && bin_high > client_low) {
                        GHPair &dest = merged_hist_data[k];
                        GHPair &src = hist_data[m - j];
                        float_type factor = (bin_high - client_low) / (client_high - client_low);
                        if (src.h != 0)
                            dest.h += src.h * factor;
                        if (src.g != 0)
                            dest.g += src.g * factor;
                    } else if (bin_low >= client_low && bin_high <= client_high) {
                        GHPair &dest = merged_hist_data[k];
                        GHPair &src = hist_data[m];
                        float_type factor = (bin_high - bin_low) / (client_high - client_low);
                        if (src.h != 0)
                            dest.h += src.h * factor;
                        if (src.g != 0)
                            dest.g += src.g * factor;
                    } else if (bin_high > client_high && bin_low < client_high) {
                        GHPair &dest = merged_hist_data[k];
                        GHPair &src = hist_data[m];
                        float_type factor = (client_high - bin_low) / (client_high - client_low);
                        if (src.h != 0)
                            dest.h += src.h * factor;
                        if (src.g != 0)
                            dest.g += src.g * factor;
                    }
                }
            }
        }
    }

    return merged_hist;
}
