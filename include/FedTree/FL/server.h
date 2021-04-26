//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_SERVER_H
#define FEDTREE_SERVER_H

#include "FedTree/FL/party.h"
#include "FedTree/dataset.h"
#include "FedTree/Tree/tree_builder.h"
//#include "FedTree/Encryption/HE.h"
#include "FedTree/DP/noises.h"
#include "FedTree/Tree/gbdt.h"
#include "omp.h"

// Todo: the server structure.

class Server : public Party {
public:
    void init(FLParam &param, int n_total_instances, vector<int> &n_instances_per_party);

    void vertical_init(FLParam &param, int n_total_instances, vector<int> &n_instances_per_party, vector<float_type> y);

    void propose_split_candidates();

    void send_info(string info_type);

    void sum_histograms();

    void hybrid_merge_trees();

    void ensemble_merge_trees();

    GBDT global_trees;
    vector<GBDT> local_trees;
    GBDTParam model_param;
    int n_total_instances;
    vector<int> n_instances_per_party;
    vector<int> n_bins_per_party;
    vector<int> n_columns_per_party;

//    AdditivelyHE::PaillierPublicKey publicKey;
//    vector<AdditivelyHE::PaillierPublicKey> pk_vector;
    Paillier paillier;

    void send_key(Party &party) {
        party.paillier = paillier;
    }

    void homo_init() {
        paillier = Paillier(512);
    }

    void decrypt_gh(GHPair &gh) {
        gh.homo_decrypt(paillier);
    }

    void decrypt_gh_pairs(SyncArray<GHPair> &encrypted) {
        auto encrypted_data = encrypted.host_data();
#pragma omp parallel for
        for (int i = 0; i < encrypted.size(); i++) {
            encrypted_data[i].homo_decrypt(paillier);
        }
    }

    void encrypt_gh_pairs(SyncArray<GHPair> &raw) {
        auto raw_data = raw.host_data();
#pragma omp parallel for
        for (int i = 0; i < raw.size(); i++) {
            raw_data[i].homo_encrypt(paillier);
        }
    }

private:
//    std::unique_ptr<TreeBuilder> fbuilder;
    DPnoises<double> DP;
};

#endif //FEDTREE_SERVER_H
