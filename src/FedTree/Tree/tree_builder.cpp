// Created by liqinbin on 11/3/20.
//

#include "FedTree/Tree/tree_builder.h"
#include "FedTree/util/cub_wrapper.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/discard_iterator.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <math.h>



void TreeBuilder::init(DataSet &dataSet, const GBDTParam &param) {
//    int n_available_device;
//    cudaGetDeviceCount(&n_available_device);
//    CHECK_GE(n_available_device, param.n_device) << "only " << n_available_device
//                                                 << " GPUs available; please set correct number of GPUs to use";
//    this->dataset = &dataSet;
    FunctionBuilder::init(dataSet, param);


    if (!dataSet.has_csc)
        dataSet.csr_to_csc();
    this->sorted_dataset = dataSet;
    seg_sort_by_key_cpu(sorted_dataset.csc_val, sorted_dataset.csc_row_idx, sorted_dataset.csc_col_ptr);
    this->n_instances = sorted_dataset.n_instances();
//    trees = vector<Tree>(1);
    ins2node_id = SyncArray<int>(n_instances);
    sp = SyncArray<SplitPoint>();
//    has_split = vector<bool>(param.n_device);
    int n_outputs = param.num_class * n_instances;
    y_predict = SyncArray<float_type>(n_outputs);
    gradients = SyncArray<GHPair>(n_instances);

}

//void TreeBuilder::split_point_all_reduce(int depth) {
//    TIMED_FUNC(timerObj);
//    //get global best split of each node
//    int n_nodes_in_level = 1 << depth;//2^i
//    int nid_offset = (1 << depth) - 1;//2^i - 1
//    auto global_sp_data = sp.host_data();
//    vector<bool> active_sp(n_nodes_in_level);
//
//
//    auto local_sp_data = sp.host_data();
//    for (int j = 0; j < sp.size(); j++) {
//        int sp_nid = local_sp_data[j].nid;
//        if (sp_nid == -1) continue;
//        int global_pos = sp_nid - nid_offset;
//        if (!active_sp[global_pos])
//            global_sp_data[global_pos] = local_sp_data[j];
//        else
//            global_sp_data[global_pos] = (global_sp_data[global_pos].gain >= local_sp_data[j].gain) ?
//                    global_sp_data[global_pos] : local_sp_data[j];
//        active_sp[global_pos] = true;
//    }
//
//    //set inactive sp
//    for (int n = 0; n < n_nodes_in_level; n++) {
//        if (!active_sp[n])
//            global_sp_data[n].nid = -1;
//    }
//
//    LOG(DEBUG) << "global best split point = " << sp;
//}

void TreeBuilder::predict_in_training(int k) {
    auto y_predict_data = y_predict.host_data() + k * n_instances;
    auto nid_data = ins2node_id.host_data();
    const Tree::TreeNode *nodes_data = trees.nodes.host_data();
    auto lr = param.learning_rate;
    #pragma omp parallel for
    for(int i = 0; i < n_instances; i++){
        int nid = nid_data[i];
        while (nid != -1 && (nodes_data[nid].is_pruned)) nid = nodes_data[nid].parent_index;
        y_predict_data[i] += lr * nodes_data[nid].base_weight;
    }
}

vector<Tree> TreeBuilder::build_approximate(const SyncArray<GHPair> &gradients, bool update_y_predict) {
    vector<Tree> trees(param.tree_per_rounds);
    TIMED_FUNC(timerObj);
    //Todo: add column sampling

    for (int k = 0; k < param.tree_per_rounds; ++k) {
        Tree &tree = trees[k];

        this->ins2node_id.resize(n_instances);
        this->gradients.set_host_data(const_cast<GHPair *>(gradients.host_data() + k * n_instances));
        this->trees.init_CPU(this->gradients, param);

        for (int level = 0; level < param.depth; ++level) {
                find_split(level);
//            split_point_all_reduce(level);
            {
                TIMED_SCOPE(timerObj, "apply sp");
                update_tree();
                update_ins2node_id();
                {
                    LOG(TRACE) << "gathering ins2node id";
                    //get final result of the reset instance id to node id
                    if (!has_split) {
                        LOG(INFO) << "no splittable nodes, stop";
                        break;
                    }
                }
//                ins2node_id_all_reduce(level);
            }
        }
        //here
        this->trees.prune_self(param.gamma);
        if(update_y_predict)
            predict_in_training(k);
        tree.nodes.resize(this->trees.nodes.size());
        tree.nodes.copy_from(this->trees.nodes);
    }
    return trees;
}

/*
 * Build a tree with the given structure and split features.
 */
void TreeBuilder::build_tree_by_predefined_structure(const SyncArray<GHPair> &gradients, vector<Tree> &trees){
    TIMED_FUNC(timerObj);
    //Todo: add column sampling

    for (int k = 0; k < param.tree_per_rounds; ++k) {
        Tree &tree = trees[k];
        this->ins2node_id.resize(n_instances);
        this->gradients.set_host_data(const_cast<GHPair *>(gradients.host_data() + k * n_instances));
        this->trees = tree;
        for (int level = 0; level < tree.final_depth; ++level) {
            LOG(INFO)<<"find split";
            find_split_by_predefined_features(level);
//            split_point_all_reduce(level);
            {
                TIMED_SCOPE(timerObj, "apply sp");
                LOG(INFO)<<"update tree";
                update_tree();
                LOG(INFO)<<"update ins2node_id";
                update_ins2node_id();
                {
                    LOG(TRACE) << "gathering ins2node id";
                    //get final result of the reset instance id to node id
                    if (!has_split) {
                        LOG(INFO) << "no splittable nodes, stop";
                        break;
                    }
                }
//                ins2node_id_all_reduce(level);
            }
        }
        //here
        this->trees.prune_self(param.gamma);
        predict_in_training(k);
        tree = this->trees;
//        tree.nodes.resize(this->trees.nodes.size());
//        tree.nodes.copy_from(this->trees.nodes);
    }
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
    auto cut_col_ptr_data = cut.cut_col_ptr.host_data();
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
        sp_data[i].split_bid = (unsigned char) (split_index % n_bins - cut_col_ptr_data[fid]);
//        sp_data[i].fea_missing_gh = missing_gh_data[i * n_column + hist_fid[split_index]];
        sp_data[i].default_right = best_split_gain < 0;
        sp_data[i].rch_sum_gh = hist_data[split_index];
    }
}

void TreeBuilder::update_tree() {
    TIMED_FUNC(timerObj);
    auto& sp = this->sp;
    auto& tree = this->trees;
    auto sp_data = sp.host_data();
    LOG(DEBUG) << sp;
    int n_nodes_in_level = sp.size();

    Tree::TreeNode *nodes_data = tree.nodes.host_data();
    float_type rt_eps = param.rt_eps;
    float_type lambda = param.lambda;

    #pragma omp parallel for
    for(int i = 0; i < n_nodes_in_level; i++){
        float_type best_split_gain = sp_data[i].gain;
        if (best_split_gain > rt_eps) {
            //do split
            //todo: check, thundergbm uses return
            if (sp_data[i].nid == -1) continue;
            if (!sp_data[i].is_change) continue;
            int nid = sp_data[i].nid;
            Tree::TreeNode &node = nodes_data[nid];
            node.gain = best_split_gain;

            Tree::TreeNode &lch = nodes_data[node.lch_index];//left child
            Tree::TreeNode &rch = nodes_data[node.rch_index];//right child
            lch.is_valid = true;
            rch.is_valid = true;
            node.split_feature_id = sp_data[i].split_fea_id;
            GHPair p_missing_gh = sp_data[i].fea_missing_gh;
            //todo process begin
            node.split_value = sp_data[i].fval;
            node.split_bid = sp_data[i].split_bid;
            rch.sum_gh_pair = sp_data[i].rch_sum_gh;
            if (sp_data[i].default_right) {
                rch.sum_gh_pair = rch.sum_gh_pair + p_missing_gh;
                node.default_right = true;
            }
            lch.sum_gh_pair = node.sum_gh_pair - rch.sum_gh_pair;
            lch.calc_weight(lambda);
            rch.calc_weight(lambda);
        } else {
            //set leaf
            //todo: check, thundergbm uses return
            if (sp_data[i].nid == -1) continue;
            int nid = sp_data[i].nid;
            Tree::TreeNode &node = nodes_data[nid];
            node.is_leaf = true;
            nodes_data[node.lch_index].is_valid = false;
            nodes_data[node.rch_index].is_valid = false;
        }
    }
    LOG(DEBUG) << tree.nodes;
}

void TreeBuilder::encrypt_gradients(AdditivelyHE::PaillierPublicKey pk) {
    auto gradients_data = gradients.host_data();
    for (int i = 0; i < gradients.size(); i++)
        gradients_data[i].homo_encrypt(pk);
}

SyncArray<GHPair> TreeBuilder::get_gradients() {
    SyncArray<GHPair> gh;
    gh.resize(gradients.size());
    gh.copy_from(gradients);
    return gh;
}

void TreeBuilder::set_gradients(SyncArray<GHPair> &gh) {
    gradients.resize(gh.size());
    gradients.copy_from(gh);
}