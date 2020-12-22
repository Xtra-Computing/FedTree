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
#include "server.h"
#include "FedTree/booster.h"

// Todo: the party structure
class Party {
public:
    void init(int pid, DataSet &dataset, FLParam &param) {
        this->pid = pid;
        this->dataset = dataset;
        booster.init(dataset, param.gbdt_param);
    };

    void homo_init() {
        std::tuple<AdditivelyHE::PaillierPublicKey, AdditivelyHE::PaillierPrivateKey> keyPairs = this->HE.generate_key_pairs();
        publicKey = std::get<0>(keyPairs);
        privateKey = std::get<1>(keyPairs);
        booster.fbuilder->encrypt_gradients(publicKey);
    };

    void overwrite_gradients(SyncArray<GHPair> gh){
        booster.fbuilder->set_gradients(gh);
    };

    void send_gradients(Party &party){
        party.overwrite_gradients(booster.fbuilder->get_gradients());
    }

    void send_last_trees(Server &server){
        server.local_trees[pid].trees[0] = this->gbdt.trees.back();
    }

    int pid;
    AdditivelyHE::PaillierPublicKey publicKey;
    AdditivelyHE::PaillierPublicKey serverKey;
//    std::unique_ptr<TreeBuilder> fbuilder;
//    vector<SplitCandidate> split_candidates;
    Booster booster;
    GBDT gbdt;
private:
    DataSet dataset;
    AdditivelyHE HE;
    AdditivelyHE::PaillierPrivateKey privateKey;
    DPnoises<double> DP;


};

#endif //FEDTREE_PARTY_H
