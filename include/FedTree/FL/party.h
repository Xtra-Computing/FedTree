//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_PARTY_H
#define FEDTREE_PARTY_H

#include "FedTree/dataset.h"
#include "FedTree/Tree/tree_builder.h"
#include "FedTree/Encryption/HE.h"
#include "FedTree/DP/noises.h"
#include "FLparam.h"
#include "FedTree/booster.h"
#include "FedTree/Tree/gbdt.h"
#include <algorithm>

// Todo: the party structure
class Party {
public:
    void init(int pid, DataSet &dataset, FLParam &param, SyncArray<bool> &feature_map) {
        this->pid = pid;
        this->dataset = dataset;
        this->param = param;
        this->feature_map.resize(feature_map.size());
        this->feature_map.copy_from(feature_map.host_data(), feature_map.size());
        booster.init(dataset, param.gbdt_param);
    };

    void send_gradients(Party &party){
        SyncArray<GHPair> gh = booster.fbuilder->get_gradients();
        if (param.privacy_tech == "dp") {
            auto gh_data = gh.host_data();
            for(int i = 0; i < gh.size(); i++) {
//                DP.add_gaussian_noise(&gh_data, param.variance);
//                gh_data[i].h = DP.add_gaussian_noise(h, param.variance);
            }
        }
        party.booster.fbuilder->set_gradients(gh);
    }

    void send_trees(Party &party) const{
        Tree tree = booster.fbuilder->get_tree();
        party.booster.fbuilder->set_tree(tree);
    }

//
//    void send_hist(Party &party){
//        SyncArray<GHPair> hist = booster.fbuilder->get_hist();
//        party.booster.fbuilder->append_hist(hist);
//    }

    void send_node(int node_id, Party &party){
        Tree::TreeNode *receiver_nodes_data = party.booster.fbuilder->trees.nodes.host_data();
        Tree::TreeNode *sender_nodes_data = booster.fbuilder->trees.nodes.host_data();
        auto& receiver_sp = party.booster.fbuilder->sp;
        auto& sender_sp = booster.fbuilder->sp;
        auto receiver_sp_data = receiver_sp.host_data();
        auto sender_sp_data = sender_sp.host_data();
        auto& receiver_ins2node_id = party.booster.fbuilder->ins2node_id;
        auto& sender_ins2node_id = booster.fbuilder->ins2node_id;
        auto receiver_ins2node_id_data = receiver_ins2node_id.host_data();
        auto sender_ins2node_id_data = sender_ins2node_id.host_data();
        int n_instances = party.booster.fbuilder->n_instances;

        receiver_nodes_data[node_id] = sender_nodes_data[node_id];
        receiver_sp_data[node_id] = sender_sp_data[node_id];

        for (int iid = 0; iid < n_instances; iid++)
            if (receiver_ins2node_id_data[iid] == node_id)
                receiver_ins2node_id_data[iid] = sender_ins2node_id_data[iid];
    }

    vector<float> slicing_value(vector<float> &val, int X, int Y) {
        auto start = val.begin() + X;
        auto end = val.begin() + Y + 1;
        vector<float> result(Y - X + 1);
        copy(start, end, result.begin());
        return result;
    }

    int get_num_feature () {
        return dataset.n_features();
    }

    // handle zero values
    float determine_min(int num_of_values, int num_of_instances, float min) {
        if (num_of_values == dataset.n_instances()) {
            return min;
        } else if (min < 0) {
            return min;
        } else return 0;
    }

    vector<float> get_feature_range_by_feature_index (int index) {
        vector<float> feature_range(2);
        vector<float> temp;

        if (index == 0) {
            int num_of_values = dataset.csc_col_ptr[0];
            if (num_of_values > 0) {
                temp = slicing_value(dataset.csc_val, 0, num_of_values - 1);
                // find max and min from temp
                auto minmax = std::minmax_element(begin(temp), end(temp));
                feature_range[1] = *minmax.second;
                feature_range[0] = determine_min(num_of_values, dataset.n_instances(), *minmax.first);
            }
        } else {
            int num_of_values = dataset.csc_col_ptr[index] - dataset.csc_col_ptr[index - 1];
            vector<float> temp = slicing_value(dataset.csc_val, dataset.csc_col_ptr[index-1], dataset.csc_col_ptr[index] - 1);
            // find max and min from temp
            auto minmax = std::minmax_element(begin(temp), end(temp));
            feature_range[1] = *minmax.second;
            feature_range[0] = determine_min(num_of_values, dataset.n_instances(), *minmax.first);
        }return feature_range;
    }

    //for hybrid fl, the parties correct the merged trees.
    void correct_trees();

    void update_tree_info();
    void compute_leaf_values();

    int pid;
    AdditivelyHE::PaillierPublicKey publicKey;
    Booster booster;
    GBDT gbdt;
    DataSet dataset;
    DPnoises<double> DP;
    FLParam param;

private:
//    AdditivelyHE HE;
//    AdditivelyHE::PaillierPrivateKey privateKey;
    SyncArray<bool> feature_map;

};

#endif //FEDTREE_PARTY_H
