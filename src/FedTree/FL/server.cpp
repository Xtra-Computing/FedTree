//
// Created by junhuiteo on 13/12/2020.
//

#include "FedTree/FL/party.h"
#include "FedTree/FL/server.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

//void send_info(vector<Party> &parties, AdditivelyHE::PaillierPublicKey serverKey,vector<SplitCandidate>candidates) {
//    for(int i = 0; i < parties.size(); i++) {
//        parties[i].serverKey = serverKey;
//        parties[i].split_candidates = candidates;
//    }
//}


void Server::merge_trees(){
    vector<Tree> &trees = global_trees.trees.back();
    int n_max_nodes_in_a_tree = pow(2, model_param.depth) - 1;
    for (int tid = 0; tid < local_trees[0].trees[0].size(); tid++) {
        // maintain an array to store the current candidates and their gains. make node_gains sorted
        // and choose the last one each time.
        vector<int> treenode_candidates;
        vector <float_type> candidate_gains;
        for (int pid = 0; pid < local_trees.size(); pid++) {
            //store the root nodes of all local trees
            //the new node id: party_id * n_max_nodes_in_a_tree + node_id
            treenode_candidates.push_back(pid * n_max_nodes_in_a_tree);
            float scale_factor = 1.0 * n_total_instances / n_instance_per_party[pid];
            candidate_gains.push_back(local_trees[pid].trees[0][tid].nodes.host_data()[0].gain * scale_factor);
            //check the node id after prune.
        }
        // sort the gain and the corresponding node ids
        thrust::stable_sort_by_key(thrust::host, candidate_gains.data(), candidate_gains.data() + candidate_gains.size(),
                                   treenode_candidates.data());
        for(int nid = 0; nid < n_max_nodes_in_a_tree; nid++) {
            int id_with_max_gain = treenode_candidates.back();
            float_type max_gain = candidate_gains.back();
            auto global_tree_node_data = trees[tid].nodes.host_data();

            int pid_max_gain = id_with_max_gain / n_max_nodes_in_a_tree;
            int nid_max_gain = id_with_max_gain % n_max_nodes_in_a_tree;
            auto node_max_gain_data = local_trees[pid_max_gain].trees[0][tid].nodes.host_data();
            global_tree_node_data[nid] = node_max_gain_data[nid_max_gain];
            global_tree_node_data[nid].gain = max_gain;
            global_tree_node_data[nid].lch_index = nid * 2 + 1;
            global_tree_node_data[nid].rch_index = nid * 2 + 2;
            global_tree_node_data[nid].parent_index = nid == 0 ? -1 : (nid - 1) / 2;
            global_tree_node_data[nid].final_id = nid;

            // pop the max element and insert the children
            treenode_candidates.pop_back();
            candidate_gains.pop_back();

            float scale_factor = 1.0 * n_total_instances / n_instance_per_party[pid_max_gain];
            int left_child = node_max_gain_data[nid_max_gain].lch_index;
            float_type left_gain = node_max_gain_data[left_child].gain * scale_factor;
            int right_child = node_max_gain_data[nid_max_gain].rch_index;
            float_type right_gain = node_max_gain_data[right_child].gain * scale_factor;
            auto insert_pos = thrust::lower_bound(thrust::host, candidate_gains.data(),
                                                  candidate_gains.data() + candidate_gains.size(), left_gain);
            auto insert_offset = insert_pos - candidate_gains.data();
            auto gain_insert_pos = candidate_gains.begin()+insert_offset;
            auto node_insert_pos = treenode_candidates.begin()+insert_offset;
            candidate_gains.insert(gain_insert_pos, node_max_gain_data[left_child].gain);
            treenode_candidates.insert(node_insert_pos, pid_max_gain * n_max_nodes_in_a_tree + left_child);
            insert_pos = thrust::lower_bound(thrust::host, candidate_gains.data(),
                                             candidate_gains.data() + candidate_gains.size(), right_gain);
            insert_offset = insert_pos - candidate_gains.data();
            gain_insert_pos = candidate_gains.begin()+insert_offset;
            node_insert_pos = treenode_candidates.begin()+insert_offset;
            candidate_gains.insert(gain_insert_pos, node_max_gain_data[right_child].gain);
            treenode_candidates.insert(node_insert_pos, pid_max_gain * n_max_nodes_in_a_tree + right_child);
        }
        //todo: add feature id offset
    }



//    // maintain an array to store the current candidates and their gains. make node_gains sorted
//    // and choose the first one each time.
//    vector<float_type> treenode_candidates;
//    vector<float_type> node_gains;
//    for(int pid = 0; pid < local_trees.size(); pid++){
//        for(int tid = 0; tid < local_trees[pid][0].size(); tid++){
//            treenode_candidates.push_back(tid);
//        }
//        treenode_candidates.push_back(tid);
//        node_gains.push_back()
//    }
//    for(int tid = 0; tid < trees.size(); tid++) {
//        Tree &tree = trees[tid];
//        tree.init_structure(model_param.depth);
//        for (int i = 0; i < model_param.depth; i++) {
//            int n_nodes_in_level = pow(2, level);
//            int nid_offset = pow(2, level) - 1;
//            for (int nid = nid_offset; nid < nid_offset + n_nodes_in_level; nid++) {
//
//                trees[nid].
//            }
//        }
//    }

}