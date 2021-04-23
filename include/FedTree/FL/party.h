//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_PARTY_H
#define FEDTREE_PARTY_H

#include "FedTree/dataset.h"
#include "FedTree/Tree/tree_builder.h"
//#include "FedTree/Encryption/HE.h"
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
        if (param.mode != "vertical" and param.mode != "horizontal") {
            this->feature_map.resize(feature_map.size());
            this->feature_map.copy_from(feature_map.host_data(), feature_map.size());
        }
        booster.init(dataset, param.gbdt_param, param.mode != "horizontal");
    };

    void send_booster_gradients(Party &party){
        SyncArray<GHPair> gh = booster.get_gradients();
        party.booster.set_gradients(gh);
    }

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


    void send_hist(Party &party){
        SyncArray<GHPair> hist = booster.fbuilder->get_hist();
        party.booster.fbuilder->append_hist(hist);
    }

    void send_node(int node_id, int n_nodes_in_level, Party &party){
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

        int lch = sender_nodes_data[node_id].lch_index;
        int rch = sender_nodes_data[node_id].rch_index;
        receiver_nodes_data[node_id] = sender_nodes_data[node_id];
        receiver_nodes_data[lch] = sender_nodes_data[lch];
        receiver_nodes_data[rch] = sender_nodes_data[rch];
        receiver_sp_data[node_id - n_nodes_in_level + 1] = sender_sp_data[node_id - n_nodes_in_level + 1];

        for (int iid = 0; iid < n_instances; iid++)
            if (receiver_ins2node_id_data[iid] == node_id)
                receiver_ins2node_id_data[iid] = sender_ins2node_id_data[iid];
    }

    int get_num_feature () {
        return dataset.n_features();
    }

    vector<float> get_feature_range_by_feature_index (int index) {
        float inf = std::numeric_limits<float>::infinity();
//        for(int i = 0; i < dataset.csr_val.size(); i++){
//            std::cout<<dataset.csr_val[i]<<" ";
//        }
        if(!dataset.has_csc)
            dataset.csr_to_csc();
        vector<float> feature_range(2);
        int column_start = dataset.csc_col_ptr[index];
        int column_end = dataset.csc_col_ptr[index+1];

        int num_of_values = column_end - column_start;
        LOG(INFO) << "feature " << index << "," << num_of_values;
        if (num_of_values > 0) {
            vector<float> temp(num_of_values);
            copy(dataset.csc_val.begin() + column_start, dataset.csc_val.begin() + column_end, temp.begin());
            LOG(INFO) << "TEMP" << temp << "feature" << index;
            auto minmax = std::minmax_element(begin(temp), end(temp));
            feature_range[1] = *minmax.second;
            feature_range[0] = *minmax.first;
        }else{
            feature_range[0] = inf;
            feature_range[1] = -inf;
        }

        return feature_range;
    }

    //for hybrid fl, the parties correct the merged trees.
    void correct_trees();

    void update_tree_info();
    void compute_leaf_values();

    int pid;
//    AdditivelyHE::PaillierPublicKey serverKey;
    Paillier paillier;

    Booster booster;
    GBDT gbdt;
    DataSet dataset;
    DPnoises<double> DP;
    FLParam param;
    int n_total_instances;

private:
//    AdditivelyHE HE;
//    AdditivelyHE::PaillierPrivateKey privateKey;
    SyncArray<bool> feature_map;

};

#endif //FEDTREE_PARTY_H
