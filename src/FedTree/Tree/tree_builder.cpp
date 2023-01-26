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
    FunctionBuilder::init(dataSet, param);

    if (!dataSet.has_csc and dataSet.csr_row_ptr.size() > 1)
        dataSet.csr_to_csc();
    this->sorted_dataset = dataSet;
    if (dataSet.csc_col_ptr.size() > 1)
        seg_sort_by_key_cpu(sorted_dataset.csc_val, sorted_dataset.csc_row_idx, sorted_dataset.csc_col_ptr);
    this->n_instances = sorted_dataset.n_instances();
    ins2node_id = SyncArray<int>(n_instances);
    sp = SyncArray<SplitPoint>();
    int n_outputs = param.num_class * n_instances;
    y_predict = SyncArray<float_type>(n_outputs);
    gradients = SyncArray<GHPair>(n_instances);
}

void TreeBuilder::init_nosortdataset(DataSet &dataSet, const GBDTParam &param) {
    FunctionBuilder::init(dataSet, param);

    if (!dataSet.has_csc and dataSet.csr_row_ptr.size() > 1)
        dataSet.csr_to_csc();
    this->sorted_dataset = dataSet;
    this->n_instances = sorted_dataset.n_instances();
    ins2node_id = SyncArray<int>(n_instances);
    sp = SyncArray<SplitPoint>();
    int n_outputs = param.num_class * n_instances;
    y_predict = SyncArray<float_type>(n_outputs);
    gradients = SyncArray<GHPair>(n_instances);
}

void TreeBuilder::set_y_predict(int k) {
    predict_in_training(k);
}

void TreeBuilder::predict_in_training(int k) {
    auto y_predict_data = y_predict.host_data() + k * n_instances;
    auto nid_data = ins2node_id.host_data();
    const Tree::TreeNode *nodes_data = trees.nodes.host_data();
    auto lr = param.learning_rate;
#pragma omp parallel for
    for (int i = 0; i < n_instances; i++) {
        int nid = nid_data[i];
        while (nid != -1 && (nodes_data[nid].is_pruned)) nid = nodes_data[nid].parent_index;
        y_predict_data[i] += lr * nodes_data[nid].base_weight;
    }
}

void TreeBuilder::build_init(const GHPair sum_gh, int k) {
    this->trees.init_CPU(sum_gh, param);
}

void TreeBuilder::build_init(const SyncArray<GHPair> &gradients, int k) {
    this->ins2node_id.resize(n_instances); // initialize n_instances here
    this->gradients.set_host_data(const_cast<GHPair *>(gradients.host_data() + k * n_instances));
    this->trees.init_CPU(this->gradients, param);
}

vector<Tree> TreeBuilder::build_approximate(const SyncArray<GHPair> &gradients, bool update_y_predict) {
    vector<Tree> trees(param.tree_per_round);
    TIMED_FUNC(timerObj);
    //Todo: add column sampling

    for (int k = 0; k < param.tree_per_round; ++k) {
        Tree &tree = trees[k];

        this->ins2node_id.resize(n_instances);
        this->gradients.set_host_data(const_cast<GHPair *>(gradients.host_data() + k * n_instances));
        this->trees.init_CPU(this->gradients, param);

        for (int level = 0; level < param.depth; ++level) {
            find_split(level);
            {
                TIMED_SCOPE(timerObj, "apply sp");
                update_tree();
                update_ins2node_id();
                {
                    LOG(TRACE) << "gathering ins2node id";
                    //get final result of the reset instance id to node id
                    if (!has_split) {
                        // LOG(INFO) << "no splittable nodes, stop";
                        break;
                    }
                }
            }
        }
        this->trees.prune_self(param.gamma);
        if (update_y_predict)
            predict_in_training(k);
        tree.nodes.resize(this->trees.nodes.size());
        tree.nodes.copy_from(this->trees.nodes);
    }
    return trees;
}

vector<Tree> TreeBuilder::build_a_subtree_approximate(const SyncArray<GHPair> &gradients, int n_layer) {
    vector<Tree> trees(param.tree_per_round);
    TIMED_FUNC(timerObj);
    //Todo: add column sampling

    for (int k = 0; k < param.tree_per_round; ++k) {
        std::cout<<"1"<<std::endl;
        Tree &tree = trees[k];

        this->ins2node_id.resize(n_instances);
        this->gradients.set_host_data(const_cast<GHPair *>(gradients.host_data() + k * n_instances));
        this->trees.init_CPU(this->gradients, param);
//        std::cout<<"1.1"<<std::endl;
        for (int level = 0; level < n_layer; ++level) {
            find_split(level);
//            std::cout<<"1.2"<<std::endl;
            {
                TIMED_SCOPE(timerObj, "apply sp");
                update_tree();
//                std::cout<<"1.3"<<std::endl;
                update_ins2node_id();
//                std::cout<<"1.4"<<std::endl;
                {
                    LOG(TRACE) << "gathering ins2node id";
                    //get final result of the reset instance id to node id
                    if (!has_split) {
                        // LOG(INFO) << "no splittable nodes, stop";
                        break;
                    }
                }
            }
        }
//        std::cout<<"2"<<std::endl;
        // prune will change tree node id, which is not consistent with ins2node_id
        // this->trees.prune_self(param.gamma);

//        std::cout<<"3"<<std::endl;
//        if (update_y_predict)
//            predict_in_training(k);
        tree.nodes.resize(this->trees.nodes.size());
        tree.nodes.copy_from(this->trees.nodes);
    }
    return trees;
}

/*
 * Build a tree with the given structure and split features.
 */
void TreeBuilder::build_tree_by_predefined_structure(const SyncArray<GHPair> &gradients, vector<Tree> &trees) {
    TIMED_FUNC(timerObj);
    //Todo: add column sampling

    for (int k = 0; k < param.tree_per_round; ++k) {
        Tree &tree = trees[k];
        this->ins2node_id.resize(n_instances);
        this->gradients.set_host_data(const_cast<GHPair *>(gradients.host_data() + k * n_instances));
        this->trees = tree;

        GHPair sum_gh = thrust::reduce(thrust::host, this->gradients.host_data(), this->gradients.host_end());

        float_type lambda = param.lambda;
        auto node_data = this->trees.nodes.host_data();
        Tree::TreeNode &root_node = node_data[0];
        root_node.sum_gh_pair = sum_gh;
        root_node.is_valid = true;
        root_node.calc_weight(lambda);
        root_node.n_instances = gradients.size();

        for (int level = 0; level < tree.final_depth; ++level) {
            LOG(INFO) << "find split";
            find_split_by_predefined_features(level);
            {
                TIMED_SCOPE(timerObj, "apply sp");
                LOG(INFO) << "update tree";
                update_tree_by_sp_values();
                LOG(INFO) << "update ins2node_id";
                update_ins2node_id();
                {
                    LOG(TRACE) << "gathering ins2node id";
                    //get final result of the reset instance id to node id
                    if (!has_split) {
                        LOG(INFO) << "no splittable nodes, stop";
                        break;
                    }
                }
            }
        }
        this->trees.prune_self(param.gamma);
        predict_in_training(k);
        tree = this->trees;
    }
}

// Remove SyncArray<GHPair> missing_gh, int n_columnf
void
TreeBuilder::find_split(SyncArray<SplitPoint> &sp, int n_nodes_in_level, Tree tree, SyncArray<int_float> best_idx_gain,
                        int nid_offset, HistCut cut, SyncArray<GHPair> hist, int n_bins) {
    sp.resize(n_nodes_in_level);
    const int_float *best_idx_gain_data = best_idx_gain.host_data();
    auto hist_data = hist.host_data();
    auto cut_val_data = cut.cut_points_val.host_data();
    auto sp_data = sp.host_data();
    auto nodes_data = tree.nodes.host_data();
    int column_offset = 0;
    auto cut_fid_data = cut.cut_fid.host_data();
    auto cut_col_ptr_data = cut.cut_col_ptr.host_data();
    auto i2fid = [=](int i) { return cut_fid_data[i % n_bins]; };
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
        sp_data[i].default_right = best_split_gain < 0;
        sp_data[i].rch_sum_gh = hist_data[split_index];
    }
}

void TreeBuilder::update_tree() {
    TIMED_FUNC(timerObj);
    auto &sp = this->sp;
    auto &tree = this->trees;
    auto sp_data = sp.host_data();
    int n_nodes_in_level = sp.size();

    Tree::TreeNode *nodes_data = tree.nodes.host_data();
    float_type rt_eps = param.rt_eps;
    float_type lambda = param.lambda;

#pragma omp parallel for
    for (int i = 0; i < n_nodes_in_level; i++) {
        float_type best_split_gain = sp_data[i].gain;
        if (best_split_gain > rt_eps) {
            //todo: check, thundergbm uses return
            if (sp_data[i].nid == -1) continue;
            int nid = sp_data[i].nid;
            Tree::TreeNode &node = nodes_data[nid];
            node.gain = best_split_gain;

            Tree::TreeNode &lch = nodes_data[node.lch_index];//left child
            Tree::TreeNode &rch = nodes_data[node.rch_index];//right child
            lch.is_valid = true; //TODO: broadcast lch and rch
            rch.is_valid = true;
            node.split_feature_id = sp_data[i].split_fea_id;
            GHPair p_missing_gh = sp_data[i].fea_missing_gh;
            //todo process begin
            node.split_value = sp_data[i].fval;
            node.split_bid = sp_data[i].split_bid;
            rch.sum_gh_pair = sp_data[i].rch_sum_gh;
            if (sp_data[i].default_right) {
                rch.sum_gh_pair = rch.sum_gh_pair + p_missing_gh;
               // LOG(INFO) << "RCH" << rch.sum_gh_pair;
                node.default_right = true;
            }
            lch.sum_gh_pair = node.sum_gh_pair - rch.sum_gh_pair;
          //  LOG(INFO) << "LCH" << lch.sum_gh_pair;
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
}

void TreeBuilder::update_tree_in_a_node(int node_id) {
    TIMED_FUNC(timerObj);
    auto &sp = this->sp;
    auto &tree = this->trees;
    auto sp_data = sp.host_data();
    int n_nodes_in_level = sp.size();

    Tree::TreeNode *nodes_data = tree.nodes.host_data();
    float_type rt_eps = param.rt_eps;
    float_type lambda = param.lambda;

    float_type best_split_gain = sp_data[node_id].gain;
    if (best_split_gain > rt_eps) {
        //do split
        if (sp_data[node_id].nid == -1) return;
        int nid = sp_data[node_id].nid;
        Tree::TreeNode &node = nodes_data[nid];
        node.gain = best_split_gain;

        Tree::TreeNode &lch = nodes_data[node.lch_index];//left child
        Tree::TreeNode &rch = nodes_data[node.rch_index];//right child
        lch.is_valid = true;
        rch.is_valid = true;
        node.split_feature_id = sp_data[node_id].split_fea_id;
        GHPair p_missing_gh = sp_data[node_id].fea_missing_gh;
        //todo process begin
        node.split_value = sp_data[node_id].fval;
        node.split_bid = sp_data[node_id].split_bid;
        rch.sum_gh_pair = sp_data[node_id].rch_sum_gh;
        if (sp_data[node_id].default_right) {
            rch.sum_gh_pair = rch.sum_gh_pair + p_missing_gh;
            node.default_right = true;
        }
        lch.sum_gh_pair = node.sum_gh_pair - rch.sum_gh_pair;
        lch.calc_weight(lambda); // TODO: check here
        rch.calc_weight(lambda); // TODO: check here
    } else {
        //set leaf
        //todo: check, thundergbm uses return
        if (sp_data[node_id].nid == -1) return;
        int nid = sp_data[node_id].nid;
        Tree::TreeNode &node = nodes_data[nid];
        node.is_leaf = true;
        nodes_data[node.lch_index].is_valid = false;
        nodes_data[node.rch_index].is_valid = false;
    }
}


void TreeBuilder::update_tree_by_sp_values() {
    TIMED_FUNC(timerObj);
    auto &sp = this->sp;
    auto &tree = this->trees;
    auto sp_data = sp.host_data();
    LOG(DEBUG) << sp;
    int n_nodes_in_level = sp.size();

    Tree::TreeNode *nodes_data = tree.nodes.host_data();
    float_type rt_eps = param.rt_eps;
    float_type lambda = param.lambda;

#pragma omp parallel for
    for (int i = 0; i < n_nodes_in_level; i++) {
        if (sp_data[i].no_split_value_update) {
            int nid = sp_data[i].nid;
            Tree::TreeNode &node = nodes_data[nid];
            Tree::TreeNode &lch = nodes_data[node.lch_index];//left child
            Tree::TreeNode &rch = nodes_data[node.rch_index];//right child

            GHPair p_missing_gh = sp_data[i].fea_missing_gh;
            //todo process begin

            rch.sum_gh_pair = sp_data[i].rch_sum_gh;
            if (sp_data[i].default_right) {
                rch.sum_gh_pair = rch.sum_gh_pair + p_missing_gh;
                node.default_right = true;
            }
            lch.sum_gh_pair = node.sum_gh_pair - rch.sum_gh_pair;
            lch.calc_weight(lambda);
            rch.calc_weight(lambda);
        } else {
            float_type best_split_gain = sp_data[i].gain;
            if (best_split_gain > rt_eps) {
                //do split
                //todo: check, thundergbm uses return
                if (sp_data[i].nid == -1) continue;
                int nid = sp_data[i].nid;
                Tree::TreeNode &node = nodes_data[nid];
                node.gain = best_split_gain;

                Tree::TreeNode &lch = nodes_data[node.lch_index];//left child
                Tree::TreeNode &rch = nodes_data[node.rch_index];//right child

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
    }
    LOG(DEBUG) << tree.nodes;
}


//void TreeBuilder::encrypt_gradients(AdditivelyHE::PaillierPublicKey pk) {
//    auto gradients_data = gradients.host_data();
//    for (int i = 0; i < gradients.size(); i++)
//        gradients_data[i].homo_encrypt(pk);
//}
//
//SyncArray<GHPair> TreeBuilder::get_gradients() {
//    SyncArray<GHPair> gh;
//    gh.resize(gradients.size());
//    gh.copy_from(gradients);
//    return gh;
//}
//
//void TreeBuilder::set_gradients(SyncArray<GHPair> &gh) {
//    gradients.resize(gh.size());
//    gradients.copy_from(gh);
//}

