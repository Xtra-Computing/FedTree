//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_PARTY_H
#define FEDTREE_PARTY_H

#include "FedTree/dataset.h"
#include "FedTree/Tree/tree_builder.h"
#include "FedTree/Encryption/HE.h"
#include "FedTree/DP/noises.h"

// Todo: the party structure
class Party {
public:
    void init(int pid, DataSet &dataset, FLParam &param, SyncArray<bool> &feature_map) {
        this->pid = pid;
        this->dataset = dataset;
        this->feature_map.resize(feature_map.size());
        this->feature_map.copy_from(feature_map.host_data(), feature_map.size());
        booster.init(dataset, param.gbdt_param);
    };

    void send_gradients(Party &party){
        SyncArray<GHPair> gh = booster.get_gradients();
        party.booster.set_gradients(gh);
    }

    void send_hist(Party &party){
        SyncArray<GHPair> hist = booster.fbuilder->get_hist();
        party.booster.fbuilder->append_hist(hist);
    }


    //for hybrid fl, the parties correct the merged trees.
    void correct_trees();

    void update_tree_info();
    void compute_leaf_values();

    int pid;

    AdditivelyHE::PaillierPublicKey publicKey;

private:
    DataSet dataset;
    AdditivelyHE HE;
    AdditivelyHE::PaillierPrivateKey privateKey;
    SyncArray<bool> feature_map;
    DPnoises<double> DP;
};

#endif //FEDTREE_PARTY_H
