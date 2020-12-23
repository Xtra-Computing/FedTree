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
        booster.init(dataset, param.gbdt_param);
    };

    void send_gradients(Party &party){
        SyncArray<GHPair> gh = booster.fbuilder->get_gradients();
        party.booster.fbuilder->set_gradients(gh);
    }

    int pid;
//    AdditivelyHE::PaillierPublicKey publicKey;
    AdditivelyHE::PaillierPublicKey serverKey;
//    std::unique_ptr<TreeBuilder> fbuilder;
//    vector<SplitCandidate> split_candidates;
    Booster booster;
    GBDT gbdt;
private:
    DataSet dataset;
//    AdditivelyHE HE;
//    AdditivelyHE::PaillierPrivateKey privateKey;
    DPnoises<double> DP;


};

#endif //FEDTREE_PARTY_H
