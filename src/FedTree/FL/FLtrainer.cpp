//
// Created by liqinbin on 10/14/20.
//
#include "FedTree/FL/FLtrainer.h"
#include "FedTree/Encryption/HE.h"
#include "FedTree/FL/partition.h"
#include "FedTree/FL/comm_helper.h"
#include "thrust/sequence.h"
#include <limits>

using namespace thrust;

void FLtrainer::horizontal_fl_trainer(vector<Party> &parties, Server &server, FLParam &params) {

    auto model_param = params.gbdt_param;

    if (params.privacy_tech == "he") {
        LOG(INFO) << "Start HE Init";
        // server generate public key and private key
        server.homo_init();
        // server distribute public key to rest of parties
        for (int i = 0; i < parties.size(); i++) {
            parties[i].publicKey = server.publicKey;
        }
        LOG(INFO) << "End of HE init";
    }

    for (int i = 0; i < params.gbdt_param.n_trees; i++) {
        // update gradients for all parties

        for (int j = 0; j < parties.size(); j++) {
            LOG(INFO) << "Party upgrade gradient";
            parties[j].booster.update_gradients();
            if (params.privacy_tech == "he") {
                LOG(INFO) << "Encrypt gradient";
                parties[i].booster.encrypt_gradients(parties[i].publicKey);
            }else if (params.privacy_tech == "dp") {
                LOG(INFO) << "Add DP noises to gradient";
                parties[i].booster.add_noise_to_gradients(params.variance);
            }
        }

        // for each tree per round
        for (int k = 0; k < params.gbdt_param.tree_per_rounds; k++) {

            // for each level
            LOG(INFO) << "Level " << k;
            for (int d = 0; d < params.gbdt_param.depth; d++) {
                LOG(INFO) << "Depth " << d;
                // initialize level parameters
                int n_nodes_in_level = 1 << d;
                int n_bins = model_param.max_num_bin;
                int n_max_nodes = 2 << model_param.depth;
                int n_max_splits = n_max_nodes * n_bins;
                MSyncArray<int> parties_hist_fid(parties.size());

                // Generate HistCut by server or each party
                if (params.propose_split == "server") {

                    // loop through all party to find max and min for each feature
                    float inf = std::numeric_limits<float>::infinity();
                    vector <vector<float>> feature_range(parties[0].get_num_feature());
                    for (int n = 0; n < parties[0].get_num_feature(); n++) {
                        vector<float> min_max = {inf, -inf};
                        for (int p = 0; p < parties.size(); p++) {
                            vector<float> temp = parties[p].get_feature_range_by_feature_index(n);
                            if (temp[0] <= min_max[0])
                                min_max[0] = temp[0];
                            if (temp[1] >= min_max[1])
                                min_max[1] = temp[1];
                        }
                        feature_range[n] = min_max;
                    }
                    LOG(INFO) << "Feature range: " << feature_range;
                    // once we have feature_range, we can generate cut points
                    for (int p = 0; p < parties.size(); p++) {
                        parties[p].booster.fbuilder->cut.get_cut_points_by_feature_range(feature_range, n_bins);
                    }
                } else if (params.propose_split == "client") {

                    for (int p = 0; p < parties.size(); p++) {
                        auto dataset = parties[p].dataset;
                        parties[p].booster.fbuilder->cut.get_cut_points_fast(dataset, n_bins, dataset.n_instances());
                    }
                }
                LOG(INFO) << "Finish Generate Cut Points";

                // Each Party Compute Histogram
                // each party compute hist, send hist to server or party
                LOG(INFO) << "Start Compute Histogram";
                Party &aggregator = (params.merge_histogram == "client")? parties[0] : server;
                LOG(INFO) << "Init Parties Hist, " << aggregator.booster.fbuilder;
                aggregator.booster.fbuilder->parties_hist_init(parties.size());
                LOG(INFO) << "Finish Parties Hist Init";

                for (int j = 0; j < parties.size(); j++) {
                    int n_column = parties[j].dataset.n_features();
                    int n_partition = n_column * n_nodes_in_level;
                    HistCut cut = parties[j].booster.fbuilder->get_cut();
                    auto cut_fid_data = cut.cut_fid.host_data();
                    SyncArray<int> hist_fid(n_nodes_in_level * n_bins);
                    auto hist_fid_data = hist_fid.host_data();

                    for (int i = 0; i < hist_fid.size(); i++)
                        hist_fid_data[i] = cut_fid_data[i % n_bins];

                    parties_hist_fid[j].resize(n_nodes_in_level * n_bins);
                    parties_hist_fid[j].copy_from(hist_fid);

                    SyncArray <GHPair> missing_gh(n_partition);
                    SyncArray <GHPair> hist(n_max_splits);
                    parties[j].booster.fbuilder->compute_histogram_in_a_level(d, n_max_splits, n_bins,
                                                                              n_nodes_in_level,
                                                                              hist_fid_data, missing_gh, hist);
                    aggregator.booster.fbuilder->append_hist(hist, missing_gh, n_partition, n_max_splits);
                }
                LOG(INFO) << "End Compute Histogram";

                // Now we have the array of hist and missing_gh
                LOG(INFO) << "Start Merging Histogram";
                if (params.propose_split == "server")
                    aggregator.booster.fbuilder->merge_histograms_server_propose();
                else if (params.propose_split == "client")
                    // TODO: Fix this to make use of missing_gh
                    aggregator.booster.fbuilder->merge_histograms_client_propose();
                // send merged histogram to server
                SyncArray <GHPair> last_hist = aggregator.booster.fbuilder->get_last_hist();
                server.booster.fbuilder->set_last_hist(last_hist);
                LOG(INFO) << "Finish Merging Histogram";

                // server compute gain
                LOG(INFO) << "Start Computing Gain";
                SyncArray <float_type> gain(n_max_splits * parties.size());
                // if privacy tech == 'he', decrypt histogram
                if (params.privacy_tech == "he")
                    server.decrypt_histogram();

                last_hist = server.booster.fbuilder->get_last_hist();
                SyncArray <GHPair> last_missing_gh = aggregator.booster.fbuilder->get_last_missing_gh();
                // if server propose cut, hist_fid for each party should be the same
                auto hist_fid_data = parties_hist_fid[0].host_data();
                server.booster.fbuilder->compute_gain_in_a_level(gain, n_nodes_in_level, n_bins, hist_fid_data,
                                                                 last_missing_gh, last_hist);
                // server find the best gain and its index
                SyncArray <int_float> best_idx_gain(n_nodes_in_level);
                server.booster.fbuilder->get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins);
                auto best_idx_data = best_idx_gain.host_data();
                LOG(INFO) << "End Computing Gain";
                LOG(INFO) << "Start Computing Split Points";
                server.booster.fbuilder->get_split_points(best_idx_gain, n_nodes_in_level, hist_fid_data, last_missing_gh,
                                                          last_hist);
                LOG(INFO) << "End Computing Split Points";
                LOG(INFO) << "Start Updating Tree";
                server.booster.fbuilder->update_tree();

                // TODO: Update trees of every party
                for (int j = 0; j < parties.size(); j++) {
                    Tree tree = parties[j].booster.fbuilder->get_tree();
                    parties[j].booster.fbuilder->set_tree(tree);
                }
                LOG(INFO) << "End Updating Tree";
            }
        }
        LOG(INFO) << server.booster.fbuilder->get_y_predict();
        LOG(INFO) << server.booster.metric->get_name() << " = "
                  << server.booster.metric->get_score(server.booster.fbuilder->get_y_predict());
    } LOG(INFO) << "end of training";
}


    void FLtrainer::vertical_fl_trainer(vector<Party> &parties, Server &server, FLParam &params) {

        // load dataset
        GBDTParam &model_param = params.gbdt_param;
        Comm comm_helper;

        // start training
        // for each boosting round
        for (int round = 0; round < params.gbdt_param.n_trees; round++) {

            // Server update, encrypt and send gradients
//        server.booster.update_gradients();
//        if (params.privacy_tech == "he")
//            server.booster.encrypt_gradients(server.publicKey);
//        else if (params.privacy_tech == "dp")
//            server.booster.add_noise_to_gradients(params.variance);
//        for (int j = 0; j < parties.size(); j++) {
//            server.send_gradients(parties[j]);
//        }

            vector<Tree> trees(params.gbdt_param.tree_per_rounds);
            server.booster.update_gradients();
            for (int pid = 0; pid < parties.size(); pid++)
                parties[pid].booster.update_gradients();

            // for each tree in a round
            for (int t = 0; t < params.gbdt_param.tree_per_rounds; t++) {
                Tree &tree = trees[t];
                // each party initialize ins2node_id, gradients, etc.
                server.booster.fbuilder->build_init(server.booster.gradients, t);
                for (int pid = 0; pid < parties.size(); pid++)
                    parties[pid].booster.fbuilder->build_init(parties[pid].booster.gradients, t);

                // for each level
                for (int l = 0; l < params.gbdt_param.depth; l++) {

                    // initialize level parameters
                    int n_nodes_in_level = 1 << l;
                    int n_max_nodes = 2 << model_param.depth;
                    vector<int> parties_n_bins(parties.size());
                    vector<int> parties_n_columns(parties.size());
                    MSyncArray<GHPair> parties_missing_gh(parties.size());
                    MSyncArray<int> parties_hist_fid(parties.size());
                    MSyncArray<GHPair> parties_hist(parties.size());

                    // each party compute hist, send hist to server
                    for (int pid = 0; pid < parties.size(); pid++) {
                        int n_bins = parties[pid].booster.fbuilder->cut.cut_points_val.size();
                        parties_n_bins[pid] = n_bins;
                        int n_max_splits = n_max_nodes * n_bins;
                        int n_column = parties[pid].dataset.n_features();
                        parties_n_columns[pid] = n_column;
                        int n_partition = n_column * n_nodes_in_level;
                        auto cut_fid_data = parties[pid].booster.fbuilder->cut.cut_fid.host_data();

                        SyncArray<int> hist_fid(n_nodes_in_level * n_bins);
                        auto hist_fid_data = hist_fid.host_data();

                        for (int i = 0; i < hist_fid.size(); i++)
                            hist_fid_data[i] = cut_fid_data[i % n_bins];

                        parties_hist_fid[pid].resize(n_nodes_in_level * n_bins);
                        parties_hist_fid[pid].copy_from(hist_fid);

                        SyncArray<GHPair> missing_gh(n_partition);
                        SyncArray<GHPair> hist(n_max_splits);
                        parties[pid].booster.fbuilder->compute_histogram_in_a_level(l, n_max_splits, n_bins,
                                                                                    n_nodes_in_level,
                                                                                    hist_fid_data, missing_gh, hist);

                        parties_missing_gh[pid].resize(n_partition);
                        parties_missing_gh[pid].copy_from(missing_gh);
                        parties_hist[pid].resize(n_max_splits);
                        parties_hist[pid].copy_from(hist);

                        parties[pid].booster.fbuilder->sp.resize(n_nodes_in_level);
                    }

                    // server concat hist_fid_data, missing_gh & histograms
                    int n_max_splits_new =
                            n_max_nodes * (*max_element(parties_n_bins.begin(), parties_n_bins.end())) * parties.size();
                    int n_bins_new = accumulate(parties_n_bins.begin(), parties_n_bins.end(), 0);
                    int n_column_new = accumulate(parties_n_columns.begin(), parties_n_columns.end(), 0);
                    SyncArray<int> hist_fid(n_bins_new * n_nodes_in_level);
                    SyncArray<GHPair> missing_gh(n_column_new * n_nodes_in_level);
                    SyncArray<GHPair> hist(n_bins_new * n_nodes_in_level);

                    hist_fid.copy_from(comm_helper.concat_msyncarray(parties_hist_fid, parties_n_bins, n_nodes_in_level));
                    missing_gh.copy_from(
                            comm_helper.concat_msyncarray(parties_missing_gh, parties_n_columns, n_nodes_in_level));
                    hist.copy_from(comm_helper.concat_msyncarray(parties_hist, parties_n_bins, n_nodes_in_level));

                    // server compute gain
                    SyncArray<float_type> gain(n_max_splits_new);

                    auto hist_fid_data = hist_fid.host_data();
                    server.booster.fbuilder->compute_gain_in_a_level(gain, n_nodes_in_level, n_bins_new, hist_fid_data,
                                                                     missing_gh, hist, n_column_new);
                    // server find the best gain and its index
                    SyncArray<int_float> best_idx_gain(n_nodes_in_level);
                    server.booster.fbuilder->get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins_new);
                    auto best_idx_data = best_idx_gain.host_data();

                    // parties who propose the best candidate update their trees accordingly
                    vector<vector<int>> party_node_map(parties.size());
                    for (int node = 0; node < n_nodes_in_level; node++) {

                        // convert the global best index to party id & its local index
                        int best_idx = get < 0 > (best_idx_data[node]);
                        float best_gain = get < 1 > (best_idx_data[node]);
                        int party_id = 0;
                        while (best_idx >= 0) {
                            best_idx -= parties_n_bins[party_id];
                            party_id += 1;
                        }
                        party_id -= 1;
                        int local_idx = best_idx + parties_n_bins[party_id];
                        party_node_map[party_id].push_back(node);

                        // party get local split point
                        parties[party_id].booster.fbuilder->get_split_points_in_a_node(node, local_idx, best_gain,
                                                                                       n_nodes_in_level,
                                                                                       parties_hist_fid[party_id].host_data(),
                                                                                       parties_missing_gh[party_id],
                                                                                       parties_hist[party_id]);

                        // party update itself
                        parties[party_id].booster.fbuilder->update_tree_in_a_node(node);
                        parties[party_id].booster.fbuilder->update_ins2node_id_in_a_node(node);
                    }

                    // party broadcast new instance space to others
                    for (int sender_id = 0; sender_id < parties.size(); sender_id++) {
                        for (int receiver_id = 0; receiver_id < parties.size(); receiver_id++) {
                            if (sender_id != receiver_id) {
                                for (int nid: party_node_map[sender_id]) {
                                    parties[sender_id].send_node(nid, parties[receiver_id]);
                                }
                            }
                        }
                    }

                    bool split_further = true;
                    for (int pid = 0; pid < parties.size(); pid++) {
                        if (!parties[pid].booster.fbuilder->has_split) {
                            split_further = false;
                            break;
                        }
                    }
                    if (!split_further)
                        break;
                }

                for (int pid = 0; pid < parties.size(); pid++) {
                    parties[pid].booster.fbuilder->trees.prune_self(model_param.gamma);
                    parties[pid].booster.fbuilder->predict_in_training(t);
                }
                tree.nodes.resize(parties[0].booster.fbuilder->trees.nodes.size());
                tree.nodes.copy_from(parties[0].booster.fbuilder->trees.nodes);
            }

            parties[0].gbdt.trees.push_back(trees);
            LOG(INFO) << parties[0].booster.fbuilder->get_y_predict();
            LOG(INFO) << parties[0].booster.metric->get_name() << " = "
                      << parties[0].booster.metric->get_score(parties[0].booster.fbuilder->get_y_predict());
        }LOG(INFO) << "end of training";
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