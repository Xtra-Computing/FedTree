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
    void init(int pid, const DataSet &dataSet) {
        this->pid = pid;
        this->dataset = dataset;
    };

    void homo_init() {
        std::tuple<AdditivelyHE::PaillierPublicKey, AdditivelyHE::PaillierPrivateKey> keyPairs = this->HE.generate_key_pairs();
        publicKey = std::get<0>(keyPairs);
        privateKey = std::get<1>(keyPairs);
        fbuilder->encrypt_gradients(publicKey);
    };

    void overwrite_gradients(SyncArray<GHPair> gh){
        fbuilder->set_gradients(gh);
    };

    void send_gradients(Party &party){
        party.overwrite_gradients(fbuilder->get_gradients());
    }

    void send_histogram(Server server) {
        server.merge_histogram()
    }

    void send_histogram(Party leader) {
        leader.merge_histgoram()
    }

    int pid;
    AdditivelyHE::PaillierPublicKey publicKey;
    AdditivelyHE::PaillierPublicKey serverKey;
    std::unique_ptr<TreeBuilder> fbuilder;
    HistCut histCut;
    SyncArray<GHPair> histogram;
//    vector<SplitCandidate> split_candidates;

private:
    DataSet dataset;
    AdditivelyHE HE;
    AdditivelyHE::PaillierPrivateKey privateKey;
    DPnoises<double> DP;
    MSyncArray<GHPair> histograms;
};

#endif //FEDTREE_PARTY_H
