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
//    void init(FLParam &param, int n_total_instances, vector<int> &n_instances_per_party);

    void horizontal_init (FLParam &param);

    void vertical_init(FLParam &param, int n_total_instances, vector<int> &n_instances_per_party, vector<float_type> y,
                       vector<float_type> label);

    void propose_split_candidates();
    void send_info(string info_type);
//    void send_info(vector<Party> &parties, AdditivelyHE::PaillierPublicKey serverKey,vector<SplitCandidate>candidates);
    void sum_histograms();
    void hybrid_merge_trees();
    void ensemble_merge_trees();

    void sample_data();
    void predict_raw_vertical_jointly_in_training(const GBDTParam &model_param, vector<Party> &parties,
                                                  SyncArray<float_type> &y_predict);
    GBDT global_trees;
    vector<GBDT> local_trees;
    GBDTParam model_param;
    vector<int> n_instances_per_party;
    

//    AdditivelyHE::PaillierPublicKey publicKey;
//    vector<AdditivelyHE::PaillierPublicKey> pk_vector;

#ifdef USE_CUDA
    Paillier_GPU paillier;
#else
    Paillier paillier;
#endif

    void send_key(Party &party) {
        party.paillier = paillier;
    }

    void homo_init() {
#ifdef USE_CUDA
        paillier.keygen();
//        pailler_gmp = Pailler(1024);
//        paillier = Paillier(paillier_gmp);
//        paillier.keygen();
#else
        paillier.keygen(512);
#endif
    }

    void decrypt_gh(GHPair &gh) {
#ifdef USE_CUDA
//        paillier.decrypt(gh);

//        mpz_t g_dec, h_dec;
//        mpz_init(g_dec);
//        mpz_init(h_dec);
//        paillier.paillier_cpu..decrypt(g_dec, g_enc);
//        paillier.paillier_cpu..decrypt(h_dec, h_enc);
//        unsigned long g_ul, h_ul;
//        mpz_export(&g_ul, 0, -1, sizeof(g_ul), 0, 0, g_dec);
//        mpz_export(&h_ul, 0, -1, sizeof(h_ul), 0, 0, h_dec);
//        g = (float_type) g_ul / 1e6;
//        h = (float_type) h_ul / 1e6;
//        encrypted = false;

//        paillier.paillier_cpu.decrypt(gh.g, gh.g_enc);
//        paillier.paillier_cpu.decrypt(gh.h, gh.h_enc);

        gh.homo_decrypt(paillier.paillier_cpu);
#else
        gh.homo_decrypt(paillier);
#endif
    }

    void decrypt_gh_pairs(SyncArray<GHPair> &encrypted) {

#ifdef USE_CUDA
        paillier.decrypt(encrypted);
        auto encrypted_data = encrypted.host_data();
        for(int i = 0; i < encrypted.size(); i++){
            encrypted_data[i].encrypted=false;
        }

//        std::cout<<"in decrypt lambda:"<<paillier.paillier_cpu.lambda<<std::endl;
//        std::cout<<"in decrypt n_square:"<<paillier.paillier_cpu.n_square<<std::endl;
//        std::cout<<"in decrypt n:"<<paillier.paillier_cpu.n<<std::endl;
//        std::cout<<"in decrypt mu:"<<paillier.paillier_cpu.mu<<std::endl;
//        std::cout<<"in decrypt g:"<<paillier.paillier_cpu.generator<<std::endl;
//        std::cout<<"in decrypt r:"<<paillier.paillier_cpu.r<<std::endl;


//        auto encrypted_data = encrypted.host_data();
//        std::cout<<"encrypted missing_gh 0 g_enc:"<<encrypted_data[0].g_enc<<std::endl;
//        #pragma omp parallel for
//        for (int i = 0; i < encrypted.size(); i++) {
//            encrypted_data[i].homo_decrypt(paillier.paillier_cpu);
//        }
//        std::cout<<"encrypted missing_gh 0 g_enc:"<<encrypted_data[0].g_enc<<std::endl;
#else
        auto encrypted_data = encrypted.host_data();
        #pragma omp parallel for
        for (int i = 0; i < encrypted.size(); i++) {
            encrypted_data[i].homo_decrypt(paillier);
        }
#endif
    }

    void encrypt_gh_pairs(SyncArray<GHPair> &raw) {
#ifdef USE_CUDA
        paillier.encrypt(raw);
        auto raw_data = raw.host_data();
        #pragma omp parallel for
        for (int i = 0; i < raw.size(); i++) {
            raw_data[i].paillier = paillier.paillier_cpu;
            raw_data[i].encrypted = true;
        }

//        auto raw_data = raw.host_data();
//        #pragma omp parallel for
//        for (int i = 0; i < raw.size(); i++) {
//            raw_data[i].homo_encrypt(paillier.paillier_cpu);
//        }
#else
        auto raw_data = raw.host_data();
        #pragma omp parallel for
        for (int i = 0; i < raw.size(); i++) {
            raw_data[i].homo_encrypt(paillier);
        }
#endif
    }

private:
//    std::unique_ptr<TreeBuilder> fbuilder;
    DPnoises<double> DP;
};

#endif //FEDTREE_SERVER_H
