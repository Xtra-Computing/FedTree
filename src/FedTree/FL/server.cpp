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


void Server::init(FLParam &param, int n_total_instances, vector<int> &n_instances_per_party){
    this->local_trees.resize(param.n_parties);
    this->model_param = param.gbdt_param;
    this->n_total_instances = n_total_instances;
    this->n_instances_per_party = n_instances_per_party;
    this->global_trees.trees.clear();
}

void Server::horizontal_init (FLParam &param, int n_total_instances, vector<int> &n_instances_per_party, DataSet dataSet) {
    init(param, n_total_instances, n_instances_per_party);
    booster.init(dataSet, param.gbdt_param);
}

void Server::vertical_init(FLParam &param, int n_total_instances, vector<int> &n_instances_per_party, DataSet dataSet){
    this->local_trees.resize(param.n_parties);
    this->model_param = param.gbdt_param;
    this->n_total_instances = n_total_instances;
    this->n_instances_per_party = n_instances_per_party;
    this->global_trees.trees.clear();
    this->dataset = dataSet;
    booster.init(dataset, param.gbdt_param);
}


void Server::hybrid_merge_trees(){
//    LOG(INFO)<<"tree 0 nodes"<<local_trees[0].trees[0][0].nodes;
//    LOG(INFO)<<"tree 1 nodes"<<local_trees[1].trees[0][0].nodes;
    int n_tree_per_round = local_trees[0].trees[0].size();
    vector<Tree> trees(n_tree_per_round);
    int n_max_internal_nodes_in_a_tree = pow(2, model_param.depth) - 1;
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
            float scale_factor = 1.0 * n_total_instances / n_instances_per_party[pid];
            //float scale_factor = 1.0;
            candidate_gains.push_back(local_trees[pid].trees[0][tid].nodes.host_data()[0].gain * scale_factor);
            //check the node id after prune.
        }
        LOG(INFO)<<"candidate gains:"<<candidate_gains;
        // sort the gain and the corresponding node ids
        thrust::stable_sort_by_key(thrust::host, candidate_gains.data(), candidate_gains.data() + candidate_gains.size(),
                                   treenode_candidates.data());
        auto global_tree_node_data = trees[tid].nodes.host_data();
        int nid = 0;
        for(; nid < n_max_internal_nodes_in_a_tree; nid++) {
//            std::cout<<"n_max_internal_nodes"<<n_max_internal_nodes_in_a_tree<<std::endl;
//            std::cout<<"tree 0 n_nodes:"<<local_trees[0].trees[0][tid].nodes.size()<<std::endl;
//            int n_internal_nodes = 0;
//            for(int i = 0; i < local_trees[0].trees[0][tid].nodes.size(); i++){
//                if(!local_trees[0].trees[0][tid].nodes.host_data()[i].is_leaf and (local_trees[0].trees[0][tid].nodes.host_data()[i].is_valid))
//                    n_internal_nodes++;
//            }
//            std::cout<<"tree 0 internal nodes:"<<n_internal_nodes<<std::endl;
//            std::cout<<"tree 1 n_nodes:"<<local_trees[1].trees[0][tid].nodes.size()<<std::endl;
//            n_internal_nodes = 0;
//            for(int i = 0; i < local_trees[1].trees[0][tid].nodes.size(); i++){
//                if(!local_trees[1].trees[0][tid].nodes.host_data()[i].is_leaf and (local_trees[1].trees[0][tid].nodes.host_data()[i].is_valid))
//                    n_internal_nodes++;
//            }
//            std::cout<<"tree 1 internal nodes:"<<n_internal_nodes<<std::endl;
//            std::cout<<"nid:"<<nid<<std::endl;
            if(!treenode_candidates.size())
                break;
            int id_with_max_gain = treenode_candidates.back();
            float_type max_gain = candidate_gains.back();

            int pid_max_gain = id_with_max_gain / n_max_internal_nodes_in_a_tree;
            int nid_max_gain = id_with_max_gain % n_max_internal_nodes_in_a_tree;
            auto node_max_gain_data = local_trees[pid_max_gain].trees[0][tid].nodes.host_data();
            global_tree_node_data[nid] = node_max_gain_data[nid_max_gain];
            global_tree_node_data[nid].gain = max_gain;
            global_tree_node_data[nid].lch_index = nid * 2 + 1;
            global_tree_node_data[nid].rch_index = nid * 2 + 2;
            global_tree_node_data[nid].parent_index = nid == 0 ? -1 : (nid - 1) / 2;
            global_tree_node_data[nid].final_id = nid;
//            global_tree_node_dright_childata[nid].split_feature_id = node_max_gain_data.split_feature_id;

            treenode_candidates.pop_back();
            candidate_gains.pop_back();
            // todo: modify, should scale by the number of instances inside the node
            //float scale_factor = 1.0 * n_total_instances / n_instance_per_party[pid_max_gain];
            int left_child = node_max_gain_data[nid_max_gain].lch_index;
            if((!node_max_gain_data[left_child].is_leaf) and node_max_gain_data[left_child].is_valid) {
                float scale_factor = 1.0 * n_total_instances / node_max_gain_data[left_child].n_instances;
                //float scale_factor = 1.0;
                float_type left_gain = node_max_gain_data[left_child].gain * scale_factor;
                auto insert_pos = thrust::lower_bound(thrust::host, candidate_gains.data(),
                                                      candidate_gains.data() + candidate_gains.size(), left_gain);
                auto insert_offset = insert_pos - candidate_gains.data();
//                if(insert_offset == candidate_gains.size()){
//                    candidate_gains.push_back(node_max_gain_data[left_child].gain);
//                    treenode_candidates.push_back(pid_max_gain * n_max_internal_nodes_in_a_tree + left_child);
//                    std::cout << "treenode_candidates size after push back:" << treenode_candidates.size();
//                }
//                else {
                auto gain_insert_pos = candidate_gains.begin() + insert_offset;
                auto node_insert_pos = treenode_candidates.begin() + insert_offset;
                candidate_gains.insert(gain_insert_pos, left_gain);
                treenode_candidates.insert(node_insert_pos,
                                           pid_max_gain * n_max_internal_nodes_in_a_tree + left_child);
//                }
            }
            int right_child = node_max_gain_data[nid_max_gain].rch_index;
            if((!node_max_gain_data[right_child].is_leaf) and node_max_gain_data[right_child].is_valid){
                float scale_factor = 1.0 * n_total_instances / node_max_gain_data[right_child].n_instances;
                //float scale_factor = 1.0;
                float_type right_gain = node_max_gain_data[right_child].gain * scale_factor;
                auto insert_pos = thrust::lower_bound(thrust::host, candidate_gains.data(),
                                                 candidate_gains.data() + candidate_gains.size(), right_gain);
                auto insert_offset = insert_pos - candidate_gains.data();
//                if(insert_offset == candidate_gains.size()){
//                    candidate_gains.push_back(node_max_gain_data[right_child].gain);
//                    treenode_candidates.push_back(pid_max_gain * n_max_internal_nodes_in_a_tree + right_child);
//                }
//                else {
                auto gain_insert_pos = candidate_gains.begin() + insert_offset;
                auto node_insert_pos = treenode_candidates.begin() + insert_offset;
                candidate_gains.insert(gain_insert_pos, right_gain);
                treenode_candidates.insert(node_insert_pos,
                                           pid_max_gain * n_max_internal_nodes_in_a_tree + right_child);
//                }
            }
        }

        for(int i = nid; i < n_max_internal_nodes_in_a_tree; i++){
            global_tree_node_data[i].is_valid = false;
        }

        int level = 0;
        trees[tid].n_nodes_level.resize(1, 0);
        for(int i = 0; i < nid; ){
            if((i + (1<<level)) >= nid) {
                trees[tid].final_depth = level + 1;
                trees[tid].n_nodes_level.push_back(trees[tid].n_nodes_level.back()+nid - i);
                break;
            }
            trees[tid].n_nodes_level.push_back(trees[tid].n_nodes_level.back()+(1<<level));
            i += (1<<level);
            level++;
        }
        //todo: add feature id offset
    }
    //todo: no need to save previous trees in the current design
    global_trees.trees.push_back(trees);
}

void Server::ensemble_merge_trees(){
    for(int i = 0; i < local_trees.size(); i++){
        for(int j = 0; j < local_trees[i].trees.size(); j++)
            global_trees.trees.push_back(local_trees[i].trees[j]);
    }
}
