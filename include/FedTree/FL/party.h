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


    void send_hist(Party &party){
        SyncArray<GHPair> hist = booster.fbuilder->get_hist();
        party.booster.fbuilder->append_hist(hist);
    }

    //for hybrid fl, the parties correct the merged trees.
    void correct_trees();

    void update_tree_info();
    void compute_leaf_values();

    int pid;
    AdditivelyHE::PaillierPublicKey serverKey;

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
