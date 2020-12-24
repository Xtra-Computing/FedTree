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
    void init(int pid, DataSet &dataset, FLParam &param) {
        this->pid = pid;
        this->dataset = dataset;
        this->param = param;
        booster.init(dataset, param.gbdt_param);
    };

    void send_gradients(Party &party){
        SyncArray<GHPair> gh = booster.fbuilder->get_gradients();
        if (param.privacy_tech == "dp") {
            auto gh_data = gh.host_data();
            for(int i = 0; i < gh.size(); i++) {
                gh_data[i].g = DP.add_gaussian_noise(gh[i].g, param.variance);
                gh_data[i].h = DP.add_gaussian_noise(gh[i].h, param.variance);
            }
        }
        party.booster.fbuilder->set_gradients(gh);
    }

    void send_trees(Party &party) const{
        Tree tree = booster.fbuilder->get_tree();
        party.booster.fbuilder->set_tree(tree);
    }

    int pid;
    AdditivelyHE::PaillierPublicKey serverKey;

    Booster booster;
    GBDT gbdt;
private:
    DataSet dataset;
    DPnoises<double> DP;
    FLParam param;


};

#endif //FEDTREE_PARTY_H
