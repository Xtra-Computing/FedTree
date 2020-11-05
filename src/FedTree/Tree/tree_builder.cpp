//
<<<<<<< HEAD
// Created by Kelly Yung on 2020/11/3.
// Referenced from thundergbm/src/thundergbm/builder/tree_builder.cu
//

#include "FedTree/Tree/tree_builder.h"
#include "FedTree/Tree/tree.h"

// Created by liqinbin on 11/3/20.
//

#include "FedTree/Tree/tree_builder.h"
#include "FedTree/Tree/hist_tree_builder.h"

TreeBuilder *TreeBuilder::create(std::string name) {
    if (name == "hist") return new HistTreeBuilder;
    LOG(FATAL) << "unknown builder " << name;
    return nullptr;
}

//
//void TreeBuilder::update_tree(auto split_data, Tree& tree, float rt_eps, float lambda) {
//    int n_nodes_in_level = split_data.size();
//    Tree::TreeNode *nodes_data = tree.nodes;
//    for (int i = 0; i < n_nodes_in_level; i++) {
//        float best_split_gain = split_data[i].gain;
//        if (best_split_gain > rt_eps) {
//            if (split_data[i].nid == -1) return;
//            int nid = split_data[i].nid;
//            Tree::TreeNode &node = nodes_data[nid];
//            node.gain = best_split_gain;
//            // left children
//            Tree::TreeNode &lch = nodes_data[node.lch_index];
//            // right children
//            Tree::TreeNode &rch = nodes_data[node.rch_index];
//            lch.is_valid = true;
//            rch.is_valid = true;
//            node.split_feature_id = split_data[i].split_fea_id;
//            // Gradient Hessian Pair
//            GHPair p_missing_gh = split_data[i].fea_missing_gh;
//            node.split_value = split_data[i].fval;
//            node.split_bid = split_data[i].split_bid;
//            rch.sum_gh_pair = split_data[i].rch_sum_gh;
//            if (split_data[i].default_right) {
//                rch.sum_gh_pair = rch.sum_gh_pair + p_missing_gh;
//                node.default_right = true;
//            }
//            lch.sum_gh_pair = node.sum_gh_pair - rch.sum_gh_pair;
//            lch.calc_weight(lambda);
//            rch.calc_weight(lambda);
//        }else {
//            //set leaf
//            if (split_data[i].nid == -1) return;
//            int nid = split_data[i].nid;
//            Tree::TreeNode &node = nodes_data[nid];
//            node.is_leaf = true;
//            nodes_data[node.lch_index].is_valid = false;
//            nodes_data[node.rch_index].is_valid = false;
//        }
//    }
//}
