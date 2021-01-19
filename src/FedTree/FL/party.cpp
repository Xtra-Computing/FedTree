//
// Created by liqinbin on 12/19/20.
//


#include "FedTree/FL/party.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

//void Party::init(int pid, DataSet &dataset, FLParam &param){
//    this->pid = pid;
//    this->dataset = dataset;
//    booster.init(dataset, param.gbdt_param);
//}


void Party::correct_trees(){
    vector<Tree> &last_trees = gbdt.trees.back();
//    auto unique_feature_end = thrust::unique_copy(thrust::host, dataset.csr_col_idx.data(),
//                        dataset.csr_col_idx.data() + dataset.csr_col_idx.size(), unique_feature_ids.host_data());
//    int unique_len = unique_feature_end - unique_feature_ids.host_data();
    auto feature_map_data = feature_map.host_data();
    for(int i = 0; i < last_trees.size(); i++){
        Tree &tree = last_trees[i];
        auto tree_nodes = tree.nodes.host_data();
        for(int nid = 0; nid < tree.nodes.size(); nid++){
            //if the node is internal node and the party has the corresponding feature id
            if(!tree_nodes[nid].is_leaf){
                if(feature_map_data[tree_nodes[nid].split_feature_id]) {
                    // calculate gain for each possible split point
                    HistCut &cut = booster.fbuilder->cut;
                }
                else{
                    //go to next level
                }

            }
            else{

            }
        }
    }
    //send gains to the server.
}

void Party::update_tree_info(){


//    HistCut &cut = booster.fbuilder->cut;
//    vector<Tree> &last_trees = gbdt.trees.back();
//    for(int tid = 0; tid < last_trees.size(); tid++){
//        Tree &tree = last_trees[tid];
//        auto root_node = tree.nodes.host_data()[0];
//        root_node.sum_gh_pair = thrust::reduce(thrust::host, booster.fbuilder->gradients.host_data(),
//                                               booster.fbuilder->gradients.host_end());
//        int split_feature_id = root_node.split_feature_id;
//        auto csc_col_ptr = dataset.csc_col_ptr.data();
//        auto csc_val_data = dataset.csc_val.data();
//        auto cut_col_ptr = cut.cut_col_ptr.host_data();
//        auto cut_val_data = cut.cut_points_val.host_data();
//        for(int cid = csc_col_ptr[split_feature_id]; cid < csc_col_ptr[split_feature_id+1]; cid++){
//            float_type feature_value = csc_val_data[cid];
//            for(int cut_id = cut_col_ptr[cid]; cut_id < cut_col_ptr[cid+1]; cut_id++){
//                float_type cut_value = cut_val_data[cut_id];
//            }
//        }
//        for(int nid = 1; nid < tree.nodes.size(); nid++){
//            auto tree_node_data = tree.nodes.host_data()[nid];
//
//        }
//    }
}

void Party::compute_leaf_values(){

}