// Created by liqinbin on 11/3/20.
//

#include "FedTree/Tree/tree_builder.h"

#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/discard_iterator.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <math.h>



void TreeBuilder::init(const DataSet &dataset, const GBMParam &param) {
//    int n_available_device;
//    cudaGetDeviceCount(&n_available_device);
//    CHECK_GE(n_available_device, param.n_device) << "only " << n_available_device
//                                                 << " GPUs available; please set correct number of GPUs to use";
    FunctionBuilder::init(dataset, param);
    this->n_instances = dataset.n_instances();
//    trees = vector<Tree>(1);
    ins2node_id = SyncArray<int>(n_instances);
    sp = SyncArray<SplitPoint>();
//    has_split = vector<bool>(param.n_device);
    int n_outputs = param.num_class * n_instances;
    y_predict = SyncArray<float_type>(n_outputs);
    gradients = SyncArray<GHPair>(n_instances);
}



vector<Tree> TreeBuilder::build_approximate(const SyncArray<GHPair> &gradients) {
    vector<Tree> trees(param.tree_per_rounds);
    TIMED_FUNC(timerObj);
    //Todo: add column sampling

    for (int k = 0; k < param.tree_per_rounds; ++k) {
        Tree &tree = trees[k];

        this->ins2node_id.resize(n_instances);
        this->gradients.set_host_data(const_cast<GHPair *>(gradients.host_data() + k * n_instances));
        this->trees.init_CPU(this->gradients, param);

        for (int level = 0; level < param.depth; ++level) {
            //here
            find_split(level);


            split_point_all_reduce(level);
            {
                TIMED_SCOPE(timerObj, "apply sp");
                update_tree();
                update_ins2node_id();
                {
                    LOG(TRACE) << "gathering ins2node id";
                    //get final result of the reset instance id to node id
                    bool has_split = false;
                    for (int d = 0; d < param.n_device; d++) {
                        has_split |= this->has_split[d];
                    }
                    if (!has_split) {
                        LOG(INFO) << "no splittable nodes, stop";
                        break;
                    }
                }
                ins2node_id_all_reduce(level);
            }
        }
        DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
            this->trees[device_id].prune_self(param.gamma);
        });
        predict_in_training(k);
        tree.nodes.resize(this->trees.front().nodes.size());
        tree.nodes.copy_from(this->trees.front().nodes);
    }
    return trees;
}




SyncArray<float_type> TreeBuilder::gain(Tree &tree, SyncArray<GHPair> &hist, int level, int n_split) {
    SyncArray<float_type> gain(n_split);
    const Tree::TreeNode *nodes_data = tree.nodes.host_data();
    float_type mcw = this->param.min_child_weight;
    float_type l = this->param.lambda;
    int nid_offset = static_cast<int>(pow(2, level) - 1);
//    SyncArray<GHPair> missing_gh(n_partition);
//    const auto missing_gh_data = missing_gh.host_data();
    GHPair *gh_prefix_sum_data = hist.host_data();
    float_type *gain_data = gain.host_data();
    for (int i = 0; i < n_split; i++) {
        int n_bins = hist.size();
        int nid = i/n_bins + nid_offset;
        if (nodes_data[nid].is_valid) {
            GHPair father_gh = nodes_data[nid].sum_gh_pair;
//            GHPair p_missing_gh = missing_gh_data[pid];
            GHPair rch_gh = gh_prefix_sum_data[i];
            float_type left_gain = std::max(0.f, compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
            gain_data[i] = left_gain;
//          rch_gh = rch_gh + p_missing_gh;
//          float_type default_to_right_gain = std::max(0.f,
//                                                   compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
//          if (default_to_left_gain > default_to_right_gain) {
//              gain_data[i] = default_to_left_gain;
//          } else {
//              gain_data[i] = -default_to_right_gain;//negative means default split to right
//          }
        } else {
            gain_data[i] = 0;
        }
    }
    return gain;
}




TreeBuilder *TreeBuilder::create(std::string name) {
    if (name == "hist") return new HistTreeBuilder;
    LOG(FATAL) << "unknown builder " << name;
    return nullptr;
}

// Remove SyncArray<GHPair> missing_gh, int n_columnf
void TreeBuilder::find_split (SyncArray<SplitPoint> &sp, int n_nodes_in_level, Tree tree, SyncArray<int_float> best_idx_gain, int nid_offset, HistCut cut, SyncArray<GHPair> hist, int n_bins) {
    sp.resize(n_nodes_in_level);
    const int_float *best_idx_gain_data = best_idx_gain.host_data();
    auto hist_data = hist.host_data();
//    const auto missing_gh_data = missing_gh.host_data();
    auto cut_val_data = cut.cut_points_val.host_data();
    auto sp_data = sp.host_data();
    auto nodes_data = tree.nodes.host_data();
    int column_offset = 0;
    auto cut_fid_data = cut.cut_fid.host_data();
    auto cut_row_ptr_data = cut.cut_row_ptr.host_data();
    auto i2fid = [=](int i) {return cut_fid_data[i % n_bins];};
    auto hist_fid = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), i2fid);

    for (int i = 0; i < n_nodes_in_level; i++) {
        int_float bsx = best_idx_gain_data[i];
        float_type best_split_gain = thrust::get<1>(bsx);
        int split_index = thrust::get<0>(bsx);

        if (!nodes_data[i + nid_offset].is_valid) {
            sp_data[i].split_fea_id = -1;
            sp_data[i].nid = -1;
            return;
        }

        int fid = hist_fid[split_index];
        sp_data[i].split_fea_id = fid + column_offset;
        sp_data[i].nid = i + nid_offset;
        sp_data[i].gain = fabsf(best_split_gain);
        sp_data[i].fval = cut_val_data[split_index % n_bins];
        sp_data[i].split_bid = (unsigned char) (split_index % n_bins - cut_row_ptr_data[fid]);
//        sp_data[i].fea_missing_gh = missing_gh_data[i * n_column + hist_fid[split_index]];
        sp_data[i].default_right = best_split_gain < 0;
        sp_data[i].rch_sum_gh = hist_data[split_index];
    }
}

void TreeBuilder::update_tree(SyncArray<SplitPoint> sp, Tree &tree, float_type rt_eps, float_type lambda) {
    int n_nodes_in_level = sp.size();
    auto sp_data = sp.host_data();
    Tree::TreeNode *nodes_data = tree.nodes.host_data();

    for (int i = 0; i < n_nodes_in_level; i++) {
        float_type best_split_gain = sp_data[i].gain;
        if (best_split_gain > rt_eps) {
            //do split
            if (sp_data[i].nid == -1) return;
            int nid = sp_data[i].nid;
            Tree::TreeNode &node = nodes_data[nid];
            node.gain = best_split_gain;

            Tree::TreeNode &lch = nodes_data[node.lch_index];//left child
            Tree::TreeNode &rch = nodes_data[node.rch_index];//right child
            lch.is_valid = true;
            rch.is_valid = true;
            node.split_feature_id = sp_data[i].split_fea_id;
//            GHPair p_missing_gh = sp_data[i].fea_missing_gh;
            node.split_value = sp_data[i].fval;
            node.split_bid = sp_data[i].split_bid;
            rch.sum_gh_pair = sp_data[i].rch_sum_gh;
//            if (sp_data[i].default_right) {
//                rch.sum_gh_pair = rch.sum_gh_pair + p_missing_gh;
//                node.default_right = true;
//            }
            lch.sum_gh_pair = node.sum_gh_pair - rch.sum_gh_pair;
            lch.calc_weight(lambda);
            rch.calc_weight(lambda);
        } else {
            //set leaf
            if (sp_data[i].nid == -1) return;
            int nid = sp_data[i].nid;
            Tree::TreeNode &node = nodes_data[nid];
            node.is_leaf = true;
            nodes_data[node.lch_index].is_valid = false;
            nodes_data[node.rch_index].is_valid = false;
        }
    }
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