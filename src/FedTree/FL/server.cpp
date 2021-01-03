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


void Server::init(FLParam &param, int n_total_instances){
    this->local_trees.resize(param.n_parties);
    this->model_param = param.gbdt_param;
    this->n_total_instances = n_total_instances;
}


void Server::hybrid_merge_trees(){
    int n_tree_per_round = local_trees[0].trees[0].size();
    vector<Tree> trees(n_tree_per_round);
    int n_max_internal_nodes_in_a_tree = pow(2, model_param.depth) - 1;
    LOG(INFO)<<"1";
    for (int tid = 0; tid < n_tree_per_round; tid++) {
        trees[tid].init_structure(model_param.depth);
        // maintain an array to store the current candidates and their gains. make node_gains sorted
        // and choose the last one each time.
        vector<int> treenode_candidates;
        vector<float_type> candidate_gains;
        for (int pid = 0; pid < local_trees.size(); pid++) {
            //store the root nodes of all local trees
            //the new node id: party_id * n_max_internal_nodes_in_a_tree + node_id
            treenode_candidates.push_back(pid * n_max_internal_nodes_in_a_tree);
//            float scale_factor = 1.0 * n_total_instances / n_instance_per_party[pid];
            float scale_factor = 1.0;
            candidate_gains.push_back(local_trees[pid].trees[0][tid].nodes.host_data()[0].gain * scale_factor);
            //check the node id after prune.
        }
        LOG(INFO)<<"2";
        // sort the gain and the corresponding node ids
        thrust::stable_sort_by_key(thrust::host, candidate_gains.data(), candidate_gains.data() + candidate_gains.size(),
                                   treenode_candidates.data());
        LOG(INFO)<<"3";
        for(int nid = 0; nid < n_max_internal_nodes_in_a_tree; nid++) {
            std::cout<<"n_max_internal_nodes"<<n_max_internal_nodes_in_a_tree<<std::endl;
            std::cout<<"tree 0 n_nodes:"<<local_trees[0].trees[0][tid].nodes.size()<<std::endl;
            int n_internal_nodes = 0;
            for(int i = 0; i < local_trees[0].trees[0][tid].nodes.size(); i++){
                if(!local_trees[0].trees[0][tid].nodes.host_data()[i].is_leaf)
                    n_internal_nodes++;
            }
            std::cout<<"tree 0 internal nodes:"<<n_internal_nodes<<std::endl;
            std::cout<<"tree 1 n_nodes:"<<local_trees[1].trees[0][tid].nodes.size()<<std::endl;
            n_internal_nodes = 0;
            for(int i = 0; i < local_trees[1].trees[0][tid].nodes.size(); i++){
                if(!local_trees[1].trees[0][tid].nodes.host_data()[i].is_leaf)
                    n_internal_nodes++;
            }
            std::cout<<"tree 1 internal nodes:"<<n_internal_nodes<<std::endl;
            std::cout<<"nid:"<<nid<<std::endl;
            int id_with_max_gain = treenode_candidates.back();
            float_type max_gain = candidate_gains.back();
            LOG(INFO)<<"3.1";
            auto global_tree_node_data = trees[tid].nodes.host_data();
            LOG(INFO)<<"3.2";
            int pid_max_gain = id_with_max_gain / n_max_internal_nodes_in_a_tree;
            int nid_max_gain = id_with_max_gain % n_max_internal_nodes_in_a_tree;
            LOG(INFO)<<"3.5";
            std::cout<<"pid max gain:"<<pid_max_gain<<std::endl;
            auto node_max_gain_data = local_trees[pid_max_gain].trees[0][tid].nodes.host_data();
            LOG(INFO)<<"3.6";
            global_tree_node_data[nid] = node_max_gain_data[nid_max_gain];
            global_tree_node_data[nid].gain = max_gain;
            global_tree_node_data[nid].lch_index = nid * 2 + 1;
            global_tree_node_data[nid].rch_index = nid * 2 + 2;
            global_tree_node_data[nid].parent_index = nid == 0 ? -1 : (nid - 1) / 2;
            global_tree_node_data[nid].final_id = nid;

            // pop the max element and insert the children
            std::cout<<"treenode candidates size:"<<treenode_candidates.size()<<std::endl;
            treenode_candidates.pop_back();
            CHECK_GT(candidate_gains.size(), 0);
            candidate_gains.pop_back();
            LOG(INFO)<<"4";
            // todo: modify, should scale by the number of instances inside the node
            //float scale_factor = 1.0 * n_total_instances / n_instance_per_party[pid_max_gain];
            int left_child = node_max_gain_data[nid_max_gain].lch_index;
            if(!node_max_gain_data[left_child].is_leaf) {
                std::cout<<"push left child"<<std::endl;
                float scale_factor = 1.0 * n_total_instances / node_max_gain_data[left_child].n_instances;
                float_type left_gain = node_max_gain_data[left_child].gain * scale_factor;
                auto insert_pos = thrust::lower_bound(thrust::host, candidate_gains.data(),
                                                      candidate_gains.data() + candidate_gains.size(), left_gain);
                LOG(INFO)<<"4.3";
                auto insert_offset = insert_pos - candidate_gains.data();
                std::cout<<"insert offset:"<<insert_offset<<std::endl;
//                if(insert_offset == candidate_gains.size()){
//                    candidate_gains.push_back(node_max_gain_data[left_child].gain);
//                    treenode_candidates.push_back(pid_max_gain * n_max_internal_nodes_in_a_tree + left_child);
//                    std::cout << "treenode_candidates size after push back:" << treenode_candidates.size();
//                }
//                else {
                auto gain_insert_pos = candidate_gains.begin() + insert_offset;
                auto node_insert_pos = treenode_candidates.begin() + insert_offset;
                LOG(INFO) << "4.4";
                std::cout << "candidate_gains size:" << candidate_gains.size() << std::endl;
                std::cout << "insert offset:" << insert_offset << std::endl;
                candidate_gains.insert(gain_insert_pos, node_max_gain_data[left_child].gain);
                LOG(INFO) << "4.5";
                treenode_candidates.insert(node_insert_pos,
                                           pid_max_gain * n_max_internal_nodes_in_a_tree + left_child);
                std::cout << "treenode_candidates size after insert:" << treenode_candidates.size();
//                }
            }
            LOG(INFO)<<"5";
            int right_child = node_max_gain_data[nid_max_gain].rch_index;
            if(!node_max_gain_data[right_child].is_leaf) {
                std::cout<<"push right child"<<std::endl;
                float scale_factor = 1.0 * n_total_instances / node_max_gain_data[right_child].n_instances;
                float_type right_gain = node_max_gain_data[right_child].gain * scale_factor;
                auto insert_pos = thrust::lower_bound(thrust::host, candidate_gains.data(),
                                                 candidate_gains.data() + candidate_gains.size(), right_gain);
                auto insert_offset = insert_pos - candidate_gains.data();
                std::cout<<"insert offset:"<<insert_offset<<std::endl;
                if(insert_offset == candidate_gains.size()){
                    candidate_gains.push_back(node_max_gain_data[right_child].gain);
                    treenode_candidates.push_back(pid_max_gain * n_max_internal_nodes_in_a_tree + right_child);
                }
                else {
                    auto gain_insert_pos = candidate_gains.begin() + insert_offset;
                    auto node_insert_pos = treenode_candidates.begin() + insert_offset;
                    candidate_gains.insert(gain_insert_pos, node_max_gain_data[right_child].gain);
                    treenode_candidates.insert(node_insert_pos,
                                               pid_max_gain * n_max_internal_nodes_in_a_tree + right_child);
                }
            }
        }
        LOG(INFO)<<"6";
        //todo: add feature id offset
    }
    global_trees.trees.push_back(trees);
}

void Server::ensemble_merge_trees(){
    for(int i = 0; i < local_trees.size(); i++){
        for(int j = 0; j < local_trees[i].trees.size(); j++)
            global_trees.trees.push_back(local_trees[i].trees[j]);
    }
}