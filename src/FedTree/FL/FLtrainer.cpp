//
// Created by liqinbin on 10/14/20.
//
#include "FedTree/FL/FLtrainer.h"
#include "FedTree/Encryption/HE.h"
#include "FedTree/FL/partition.h"
#include "FedTree/FL/comm_helper.h"
#include "thrust/sequence.h"

using namespace thrust;

void FLtrainer::horizontal_fl_trainer(vector<Party> &parties, Server &server, FLParam &params) {
// Is propose_split_candidates implemented? Should it be a method of TreeBuilder, HistTreeBuilder or server? Shouldnt there be a vector of SplitCandidates returned
//  vector<SplitCandidate> candidates = server.fbuilder.propose_split_candidates();
//  std::tuple <AdditivelyHE::PaillierPublicKey, AdditivelyHE::PaillierPrivateKey> key_pair = server.HE.generate_key_pairs();
//    server.send_info(parties, std::get<0>(keyPairs), candidates);
//    for (int i = 0; i < params.gbdt_param.n_trees; i++){
//        for (j = 0; j < parties.size(); j++){
//            parties[j].update_gradients();
//        }
//        for (int j = 0; j < params.gbdt_param.depth; j++){
//            for (int k = 0; k < parties.size(); k++) {
//                SyncArray<GHPair> hist = parties[j].fbuilder->compute_histogram();
//                if (params.privacy_tech == "he") {
    // Should HE be a public member of Party?
//                    parties[k].HE.encryption();
//                }
//                if (params.privacy_tech == "dp") {
    // Should DP be public member of Party?
//                    parties[k].DP.add_gaussian_noise();
//                }
//                parties[k].send_info(hist);
//            }
    // merge_histograms in tree_builder?
//            server.sum_histograms(); // or on Party 1 if using homo encryption
//            server.HE.decrption();
//            if (j != params.gbdt_param.depth - 1) {
//                server.fbuilder.compute_gain();
//                server.fbuilder.get_best_split(); // or using exponential mechanism
//                server.fbuilder.update_tree();
//                server.send_info(); // send split points
//            }
//            else{
//                server.fbuilder.compute_leaf_value();
//                server.DP.add_gaussian_noise(); // for DP: add noises to the tree
//                server.send_info(); // send leaf values
//            }
//        }
//    }
}


void FLtrainer::vertical_fl_trainer(vector<Party> &parties, Server &server, FLParam &params) {

    // load dataset
    GBDTParam &model_param = params.gbdt_param;

    // start training
    // for each boosting round
    for (int i = 0; i < params.gbdt_param.n_trees; i++) {

        // Server update, encrypt and send gradients
        server.booster.update_gradients();
        if (params.privacy_tech == "he")
            server.booster.encrypt_gradients(server.publicKey);
        else if (params.privacy_tech == "dp")
            server.booster.add_noise_to_gradients(params.variance);
        for (int j = 0; j < parties.size(); j++) {
            server.send_gradients(parties[j]);
        }

        // for each tree in a round
        for (int k = 0; k < params.gbdt_param.tree_per_rounds; k++) {

            // each party initialize ins2node_id, gradients, etc.
            for (int j = 0; j < parties.size(); j++)
                parties[j].booster.fbuilder->build_init(parties[j].booster.get_gradients(), k);

            // for each level
            for (int l = 0; l < params.gbdt_param.depth; l++) {

                // initialize level parameters
                int n_nodes_in_level = 1 << l;
                int n_bins = model_param.max_num_bin;
                int n_max_nodes = 2 << model_param.depth;
                int n_max_splits = n_max_nodes * n_bins;
                MSyncArray<GHPair> parties_missing_gh(parties.size());
                MSyncArray<int> parties_hist_fid(parties.size());
                MSyncArray<GHPair> parties_hist(parties.size());

                // each party compute hist, send hist to server
                for (int j = 0; j < parties.size(); j++) {
                    int n_column = parties[j].dataset.n_features();
                    int n_partition = n_column * n_nodes_in_level;
                    HistCut cut = parties[j].booster.fbuilder->get_cut();
                    auto cut_fid_data = cut.cut_fid.host_data();

                    SyncArray<int> hist_fid(n_nodes_in_level * n_bins);
                    auto hist_fid_data = hist_fid.host_data();

#pragma omp parallel for
                    for (int i = 0; i < hist_fid.size(); i++)
                        hist_fid_data[i] = cut_fid_data[i % n_bins];

                    parties_hist_fid[j].resize(n_nodes_in_level * n_bins);
                    parties_hist_fid[j].copy_from(hist_fid);

                    SyncArray<GHPair> missing_gh(n_partition);
                    SyncArray<GHPair> hist(n_max_splits);
                    parties[j].booster.fbuilder->compute_histogram_in_a_level(l, n_max_splits, n_bins, n_nodes_in_level,
                                                                              hist_fid_data, missing_gh, hist);
                    parties_missing_gh[j].resize(n_partition);
                    parties_missing_gh[j].copy_from(missing_gh);
                    parties_hist[j].resize(n_max_splits);
                    parties_hist[j].copy_from(hist);
                }

                // server concat hist_fid_data, missing_gh & histograms
                SyncArray<int> hist_fid = concat_msyncarray(parties_hist_fid);
                SyncArray<GHPair> missing_gh = concat_msyncarray(parties_missing_gh);
                SyncArray<GHPair> hist = concat_msyncarray(parties_hist);

                // server compute gain
                SyncArray<float_type> gain(n_max_splits * parties.size());
                auto hist_fid_data = hist_fid.host_data();
                server.booster.fbuilder->compute_gain_in_a_level(gain, n_nodes_in_level, n_bins, hist_fid_data,
                                                                 missing_gh, hist);
                // server find the best gain and its index
                SyncArray<int_float> best_idx_gain(n_nodes_in_level);
                server.booster.fbuilder->get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins);
                auto best_idx_data = best_idx_gain.host_data();

                // parties who propose the best candidate update their trees accordingly
                for (int node = 0; node < n_nodes_in_level; node++) {

                    // convert the global best index to party id & its local index
                    int idx = get < 0 > (best_idx_data[node]);
                    float gain = get < 1 > (best_idx_data[node]);
                    int party_id = idx / n_max_splits;
                    int local_idx = idx % n_max_splits;

                    // party get local split point
                    parties[party_id].booster.fbuilder->get_split_points_in_a_node(node, local_idx, gain,
                                                                                   n_nodes_in_level, hist_fid_data,
                                                                                   missing_gh, hist);

                    // party update itself
//                    parties[party_id].booster.fbuilder->update_tree_in_a_node(node);
                    parties[party_id].booster.fbuilder->update_ins2node_id_in_a_node(node);

                    // party broadcast new instance space to others
//                    parties[party_id].broadcast_tree();
                }
            }
        }
    }
}

template<class T>
SyncArray<T> FLtrainer::concat_msyncarray(MSyncArray<T> &arrays) {
    int total_size = 0;
    vector<int> ptr = {0};
    for (int i = 0; i < arrays.size(); i++) {
        total_size += arrays[i].size();
        ptr.push_back(ptr.back() + total_size);
    }
    SyncArray<T> concat_array(total_size);
    auto concat_array_data = concat_array.host_data();

    for (int i = 0; i < arrays.size(); i++) {
        auto array_data = arrays[i].host_data();
        for (int j = 0; j < arrays[i].size(); j++) {
            concat_array_data[ptr[i] + j] = array_data[j];
        }
    }
    return concat_array;
}


void FLtrainer::hybrid_fl_trainer(vector<Party> &parties, Server &server, FLParam &params) {
    // todo: initialize parties and server
    int n_party = parties.size();
    Comm comm_helper;
    for (int i = 0; i < params.gbdt_param.n_trees; i++) {
        // There is already omp parallel inside boost
//        #pragma omp parallel for
        for (int pid = 0; pid < n_party; pid++) {
            LOG(INFO) << "boost without prediction";
            parties[pid].booster.boost_without_prediction(parties[pid].gbdt.trees);
//            obj->get_gradient(y, fbuilder->get_y_predict(), gradients);
            LOG(INFO) << "send last trees to server";
            comm_helper.send_last_trees_to_server(parties[pid], pid, server);
            parties[pid].gbdt.trees.pop_back();
        }
        LOG(INFO) << "merge trees";
        server.hybrid_merge_trees();
        LOG(INFO) << "send back trees";
        // todo: send the trees to the party to correct the trees and compute leaf values
//        #pragma omp parallel for
        for (int pid = 0; pid < n_party; pid++) {
            LOG(INFO) << "in party:" << pid;
            comm_helper.send_last_global_trees_to_party(server, parties[pid]);
            LOG(INFO) << "personalize trees";
            //LOG(INFO)<<"gradients before correct sps"<<parties[pid].booster.gradients;
            // todo: prune the tree, if n_bin is 0, then a half of the child tree is useless.
            parties[pid].booster.fbuilder->build_tree_by_predefined_structure(parties[pid].booster.gradients,
                                                                              parties[pid].gbdt.trees.back());
        }
    }
}

void FLtrainer::ensemble_trainer(vector<Party> &parties, Server &server, FLParam &params) {
    int n_party = parties.size();
    CHECK_EQ(params.gbdt_param.n_trees % n_party, 0);
    int n_tree_each_party = params.gbdt_param.n_trees / n_party;
    Comm comm_helper;
//    #pragma omp parallel for
    for (int i = 0; i < n_party; i++) {
        for (int j = 0; j < n_tree_each_party; j++)
            parties[i].booster.boost(parties[i].gbdt.trees);
        comm_helper.send_all_trees_to_server(parties[i], i, server);
    }
    server.ensemble_merge_trees();
}

void FLtrainer::solo_trainer(vector<Party> &parties, FLParam &params) {
    int n_party = parties.size();
//    #pragma omp parallel for
    for (int i = 0; i < n_party; i++) {
//        parties[i].gbdt.train(params.gbdt_param, parties[i].dataset);
        for (int j = 0; j < params.gbdt_param.n_trees; j++)
            parties[i].booster.boost(parties[i].gbdt.trees);
    }
}