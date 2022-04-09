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
#ifdef USE_CUDA
#include "FedTree/Encryption/paillier_gpu.h"
#endif



class Party {
public:
    void init(int pid, DataSet &dataset, FLParam &param, SyncArray<bool> &feature_map);

    void bagging_init(int seed = -1);

    void vertical_init(int pid, DataSet &dataset, FLParam &param) {
        this->pid = pid;
        this->dataset = dataset;
        this->param = param;
        this->n_total_instances = dataset.n_instances();
        booster.init(dataset, param.gbdt_param);
    };

    void send_booster_gradients(Party &party) {
        SyncArray<GHPair> gh = booster.get_gradients();
        party.booster.set_gradients(gh);
    }

    void send_gradients(Party &party) {
        SyncArray<GHPair> gh = booster.fbuilder->get_gradients();
        if (param.privacy_tech == "dp") {
            auto gh_data = gh.host_data();
            for (int i = 0; i < gh.size(); i++) {
//                DP.add_gaussian_noise(&gh_data, param.variance);
//                gh_data[i].h = DP.add_gaussian_noise(h, param.variance);
            }
        }
        party.booster.fbuilder->set_gradients(gh);
    }

    void send_trees(Party &party) const {
        Tree tree = booster.fbuilder->get_tree();
        party.booster.fbuilder->set_tree(tree);
    }


    void send_hist(Party &party) {
        SyncArray<GHPair> hist = booster.fbuilder->get_hist();
        party.booster.fbuilder->append_hist(hist);
    }

    void send_node(int node_id, int n_nodes_in_level, Party &party) {
        Tree::TreeNode *receiver_nodes_data = party.booster.fbuilder->trees.nodes.host_data();
        Tree::TreeNode *sender_nodes_data = booster.fbuilder->trees.nodes.host_data();
//        auto &receiver_sp = party.booster.fbuilder->sp;
//        auto &sender_sp = booster.fbuilder->sp;
//        auto receiver_sp_data = receiver_sp.host_data();
//        auto sender_sp_data = sender_sp.host_data();
        auto &receiver_ins2node_id = party.booster.fbuilder->ins2node_id;
        auto &sender_ins2node_id = booster.fbuilder->ins2node_id;
        auto receiver_ins2node_id_data = receiver_ins2node_id.host_data();
        auto sender_ins2node_id_data = sender_ins2node_id.host_data();
        int n_instances = party.booster.fbuilder->n_instances;

        int lch = sender_nodes_data[node_id].lch_index;
        int rch = sender_nodes_data[node_id].rch_index;
        receiver_nodes_data[node_id] = sender_nodes_data[node_id];
        receiver_nodes_data[lch] = sender_nodes_data[lch];
        receiver_nodes_data[rch] = sender_nodes_data[rch];
//        receiver_sp_data[node_id - n_nodes_in_level + 1] = sender_sp_data[node_id - n_nodes_in_level + 1];

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
        if (num_of_values > 0) {
            vector<float> temp(num_of_values);
            copy(dataset.csc_val.begin() + column_start, dataset.csc_val.begin() + column_end, temp.begin());
            auto minmax = std::minmax_element(begin(temp), end(temp));
            feature_range[1] = *minmax.second;
            feature_range[0] = *minmax.first;
        }else{
            // Does not have any value for this feature
            feature_range[0] = inf;
            feature_range[1] = -inf;
        }

        return feature_range;
    }

    void encrypt_histogram(SyncArray<GHPair> &hist) {
#ifdef USE_CUDA
        paillier.encrypt(hist);
        auto hist_data = hist.host_data();
        #pragma omp parallel for
        for(int i = 0; i < hist.size(); i++){
            hist_data[i].paillier = paillier.paillier_cpu;
//            hist_data[i].g = 0;
//            hist_data[i].h = 0;
            hist_data[i].encrypted=true;
        }

//        auto hist_data = hist.host_data();
//        #pragma omp parallel for
//        for (int i = 0; i < hist.size(); i++) {
//            hist_data[i].homo_encrypt(paillier.paillier_cpu);
//        }
#else
        auto hist_data = hist.host_data();
        #pragma omp parallel for
        for (int i = 0; i < hist.size(); i++) {
            hist_data[i].homo_encrypt(paillier);
        }
#endif
    }

//    void encrypt_gradient(GHPair &ghpair) {
//        ghpair.homo_encrypt(paillier.paillier_cpu);
//    }

    void sample_data();

    //for hybrid fl, the parties correct the merged trees.
    void correct_trees();

    void update_tree_info();

    void compute_leaf_values();

    int pid;
//    AdditivelyHE::PaillierPublicKey serverKey;
#ifdef USE_CUDA
    Paillier_GPU paillier;
#else
    Paillier paillier;
#endif
    Booster booster;
    GBDT gbdt;
    DataSet dataset;
    float ins_bagging_fraction; //store the bagging fraction
    vector<int> shuffle_idx; // store the shuffled instance IDs for bagging
    DataSet temp_dataset; //store the original dataset when do bagging
    int bagging_inner_round; //store the round number inside a bagging loop
    DPnoises<double> DP;
    FLParam param;
    int n_total_instances;

private:
//    AdditivelyHE HE;
//    AdditivelyHE::PaillierPrivateKey privateKey;
    SyncArray<bool> feature_map;

};

#endif //FEDTREE_PARTY_H
