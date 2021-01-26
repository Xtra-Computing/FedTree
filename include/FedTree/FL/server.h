//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_SERVER_H
#define FEDTREE_SERVER_H

#include "FedTree/dataset.h"
#include "FedTree/Tree/tree_builder.h"
#include "FedTree/Encryption/HE.h"
#include "FedTree/DP/noises.h"

// Todo: the server structure.

class Server {
public:
    AdditivelyHE::PaillierPublicKey publicKey;
    void homo_init() {
        std::tuple<AdditivelyHE::PaillierPublicKey, AdditivelyHE::PaillierPrivateKey> keyPairs = this->HE.generate_key_pairs();
        publicKey = std::get<0>(keyPairs);
        privateKey = std::get<1>(keyPairs);
    };

    void send_split_candidates(vector<Party> &parties) {
        for (Party p : parties) {
            p.histCut = histCut;
        }
    }
    void send_public_key(vector<Party> &parties) {
        for (Party p : parties) {
            p.publicKey = publicKey;
        }
    };
//    void send_info(vector<Party> &parties, AdditivelyHE::PaillierPublicKey serverKey,vector<SplitCandidate>candidates);
    void sum_histograms();

private:
    DataSet dataset;
    std::unique_ptr<TreeBuilder> fbuilder;
    AdditivelyHE HE;
    AdditivelyHE::PaillierPrivateKey privateKey;
    DPnoises<double> DP;
    MSyncArray<GHPair> histograms;
};

#endif //FEDTREE_SERVER_H
