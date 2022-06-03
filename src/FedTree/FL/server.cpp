//
// Created by junhuiteo on 13/12/2020.
//

#include "FedTree/FL/party.h"
#include "FedTree/FL/server.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <limits>


//void send_info(vector<Party> &parties, AdditivelyHE::PaillierPublicKey serverKey,vector<SplitCandidate>candidates) {
//    for(int i = 0; i < parties.size(); i++) {
//        parties[i].serverKey = serverKey;
//        parties[i].split_candidates = candidates;
//    }
//}


//void Server::init(FLParam &param, vector<int> &n_instances_per_party){
//    this->local_trees.resize(param.n_parties);
//    this->model_param = param.gbdt_param;
//    this->n_instances_per_party = n_instances_per_party;
//    this->global_trees.trees.clear();
//}

void Server::horizontal_init(FLParam &param) {
    DataSet dataSet;
    this->local_trees.resize(param.n_parties);
    this->model_param = param.gbdt_param;
    this->global_trees.trees.clear();
    this->has_label.resize(param.n_parties);
    booster.fbuilder.reset(new HistTreeBuilder);
    booster.fbuilder->init_nocutpoints(dataSet, param.gbdt_param); //if remove this line, cannot successfully run
    booster.param = param.gbdt_param;
}


//void Server::horizontal_init(FLParam &param, int n_total_instances, vector<int> &n_instances_per_party, DataSet& dataset) {
////    DataSet dataSet;
//    this->local_trees.resize(param.n_parties);
//    this->model_param = param.gbdt_param;
//    this->n_instances_per_party = n_instances_per_party;
//    this->global_trees.trees.clear();
//    this->n_total_instances = n_total_instances;
////    init(param, n_total_instances, n_instances_per_party);
////    booster.init(dataset, param.gbdt_param);
//    booster.fbuilder.reset(new HistTreeBuilder);
//    booster.fbuilder->init_nocutpoints(dataset, param.gbdt_param); //if remove this line, cannot successfully run
//    booster.param = param.gbdt_param;
//    booster.obj.reset(ObjectiveFunction::create(param.gbdt_param.objective));
//    booster.obj->configure(param.gbdt_param, dataSet);
//    if (param.gbdt_param.metric == "default") {
//        booster.metric.reset(Metric::create(booster.obj->default_metric_name()));
//    }else {
//        booster.metric.reset(Metric::create(param.gbdt_param.metric));
//    }
//    booster.metric->configure(param.gbdt_param, dataSet);
//    booster.n_devices = param.gbdt_param.n_device;
//    int n_outputs = param.gbdt_param.num_class * dataSet.n_instances();
//    booster.gradients.resize(n_outputs);
//    booster.y = SyncArray<float_type>(dataSet.n_instances());
//    booster.y.copy_from(dataSet.y.data(), dataSet.n_instances());
//}


void Server::vertical_init(FLParam &param, int n_total_instances, vector<int> &n_instances_per_party, vector<float_type> y,
                           vector<float_type> label){

    this->local_trees.resize(param.n_parties);
    this->model_param = param.gbdt_param;
    this->n_total_instances = n_total_instances;
    this->n_instances_per_party = n_instances_per_party;
    this->global_trees.trees.clear();
    this->has_label.resize(param.n_parties);
    dataset.y = y;
    dataset.n_features_ = 0;
    dataset.label = label;
    if (param.ins_bagging_fraction < 1.0){
        this->temp_dataset = dataset;
        this->ins_bagging_fraction = param.ins_bagging_fraction;
    }
    booster.init(dataset, param.gbdt_param);
}

void Server::sample_data(){
    int stride = this->ins_bagging_fraction * this->n_total_instances;
    vector<int> batch_idx;
    if(this->bagging_inner_round == (int(1/this->ins_bagging_fraction) - 1)){
        batch_idx = vector<int>(this->shuffle_idx.begin()+stride*this->bagging_inner_round, this->shuffle_idx.end());
    }
    else {
        batch_idx = vector<int>(this->shuffle_idx.begin() + stride * this->bagging_inner_round,
                                this->shuffle_idx.begin() + stride * (this->bagging_inner_round + 1));
    }
    std::sort(batch_idx.begin(), batch_idx.end());
    this->dataset.y.clear();
    this->dataset.y.resize(batch_idx.size());
    for(int i = 0; i < batch_idx.size(); i++)
        this->dataset.y[i] = this->temp_dataset.y[batch_idx[i]];
    this->bagging_inner_round++;
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


void Server::predict_raw_vertical_jointly_in_training(const GBDTParam &model_param, vector<Party>& parties, SyncArray<float_type> &y_predict) {
    TIMED_SCOPE(timerObj, "predict");
    int n_instances = parties[0].dataset.n_instances();
//    int n_features = dataSet.n_features();
    auto trees = global_trees.trees;
    //the whole model to an array
    int num_iter = trees.size();
    int num_class = trees.front().size();
    int num_node = trees[0][0].nodes.size();
    int total_num_node = num_iter * num_class * num_node;
    //TODO: reduce the output size for binary classification
    y_predict.resize(n_instances * num_class);

    SyncArray<Tree::TreeNode> model(total_num_node);
    auto model_data = model.host_data();
    int tree_cnt = 0;
    for (auto &vtree:trees) {
        for (auto &t:vtree) {
            memcpy(model_data + num_node * tree_cnt, t.nodes.host_data(), sizeof(Tree::TreeNode) * num_node);
            tree_cnt++;
        }
    }

    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "init trees");

    //do prediction
    auto model_host_data = model.host_data();
    auto predict_data = y_predict.host_data();

    auto lr = model_param.learning_rate;
    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "copy data");

    //predict BLOCK_SIZE instances in a block, 1 thread for 1 instance
//    int BLOCK_SIZE = 128;
    //determine whether we can use shared memory
//    size_t smem_size = n_features * BLOCK_SIZE * sizeof(float_type);
//    int NUM_BLOCK = (n_instances - 1) / BLOCK_SIZE + 1;

    vector<int> parties_n_columns(parties.size());
    for (int pid = 0; pid < parties.size(); pid++) {
        parties_n_columns[pid] = parties[pid].dataset.n_features();
    }
    //use sparse format and binary search
#pragma omp parallel for
    for (int iid = 0; iid < n_instances; iid++) {
        auto get_next_child = [&](Tree::TreeNode node, float_type feaValue) {
            //return feaValue < node.split_value ? node.lch_index : node.rch_index;
            return (feaValue - node.split_value) >= -1e-6 ? node.rch_index : node.lch_index;
        };
        auto get_val = [&](const int *row_idx, const float_type *row_val, int row_len, int idx,
                           bool *is_missing) -> float_type {
            //binary search to get feature value
            const int *left = row_idx;
            const int *right = row_idx + row_len;

            while (left != right) {
                const int *mid = left + (right - left) / 2;
                if (*mid == idx) {
                    *is_missing = false;
                    return row_val[mid - row_idx];
                }
                if (*mid > idx)
                    right = mid;
                else left = mid + 1;
            }
            *is_missing = true;
            return 0;
        };
        for (int t = 0; t < num_class; t++) {
            auto predict_data_class = predict_data + t * n_instances;
            float_type sum = 0;
            for (int iter = 0; iter < num_iter; iter++) {
                const Tree::TreeNode *node_data = model_host_data + iter * num_class * num_node + t * num_node;
                Tree::TreeNode curNode = node_data[0];
                int cur_nid = 0; //node id
                while (!curNode.is_leaf) {
                    int pid = 0;
                    int fid = curNode.split_feature_id;
                    while(parties_n_columns[pid]<fid){
                        fid -= parties_n_columns[pid];
                        pid++;
                    }
                    //conduct in Party pid
                    DataSet& data = parties[pid].dataset;
                    bool is_missing;
                    float_type fval = get_val(data.csr_col_idx.data() + data.csr_row_ptr.data()[iid],
                                              data.csr_val.data()+ data.csr_row_ptr.data()[iid],
                                              data.csr_row_ptr.data()[iid+1] - data.csr_row_ptr.data()[iid],
                                              fid, &is_missing);
                    if (!is_missing)
                        cur_nid = get_next_child(curNode, fval);
                    else if (curNode.default_right)
                        cur_nid = curNode.rch_index;
                    else
                        cur_nid = curNode.lch_index;
                    curNode = node_data[cur_nid];
                }
                sum += lr * node_data[cur_nid].base_weight;
                if (model_param.bagging)
                    sum /= num_iter;
            }
            predict_data_class[iid] += sum;
        }//end all tree prediction
    }
}
