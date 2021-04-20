//
// Created by liqinbin on 10/14/20.
//
#include "FedTree/DP/differential_privacy.h"
#include "FedTree/FL/FLtrainer.h"
#include "FedTree/Encryption/HE.h"
#include "FedTree/FL/partition.h"
#include "FedTree/FL/comm_helper.h"
#include "thrust/sequence.h"
#include "thrust/reduce.h"
#include "thrust/execution_policy.h"
#include <limits>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace thrust;

void FLtrainer::horizontal_fl_trainer(vector<Party> &parties, Server &server, FLParam &params) {
    auto model_param = params.gbdt_param;
    Party &aggregator = (params.merge_histogram == "client")? parties[0] : server;
    aggregator.booster.fbuilder->parties_hist_init(parties.size());

//    if (params.privacy_tech == "he") {
//        LOG(INFO) << "Start HE Init";
//        // server generate public key and private key
//        server.homo_init();
//        // server distribute public key to rest of parties
//        for (int i = 0; i < parties.size(); i++) {
//            parties[i].publicKey = server.publicKey;
//        }
//        LOG(INFO) << "End of HE init";
//    }

    // Generate HistCut by server or each party
    int n_bins = model_param.max_num_bin;

    if (params.propose_split == "server") {
        // loop through all party to find max and min for each feature
        float inf = std::numeric_limits<float>::infinity();
        vector<vector<float>> feature_range(parties[0].get_num_feature());
//        for(int i = 0; i < parties[0].dataset.csr_val.size(); i++)
//            std::cout<<parties[0].dataset.csr_val[i]<<" ";
        for (int n = 0; n < parties[0].get_num_feature(); n++) {
            vector<float> min_max = {inf, -inf};
            for (int p = 0; p < parties.size(); p++) {
                vector<float> temp = parties[p].get_feature_range_by_feature_index(n);
                if (temp[0] <= min_max[0] && temp[0] != -inf)
                    min_max[0] = temp[0];
                if (temp[1] >= min_max[1] && temp[1] != inf)
                    min_max[1] = temp[1];
            }
            feature_range[n] = min_max;
            LOG(INFO) << "MINMAX" << min_max;
        }
        LOG(INFO) << feature_range;
//        // once we have feature_range, we can generate cut points
        server.booster.fbuilder->cut.get_cut_points_by_feature_range(feature_range, n_bins);
        //server.booster.fbuilder->get_bin_ids();
        for (int p = 0; p < parties.size(); p++) {
            parties[p].booster.fbuilder->cut.get_cut_points_by_feature_range(feature_range, n_bins);
            parties[p].booster.fbuilder->get_bin_ids();
        }


    } else if (params.propose_split == "client") {
        for (int p = 0; p < parties.size(); p++) {
            auto dataset = parties[p].dataset;
            parties[p].booster.fbuilder->cut.get_cut_points_fast(dataset, n_bins, dataset.n_instances());
            aggregator.booster.fbuilder->append_to_parties_cut(parties[p].booster.fbuilder->cut, p);
        }
//        LOG(INFO)<<"not supported yet";
//        exit(1);
    }

    for (int i = 0; i < params.gbdt_param.n_trees; i++) {

        LOG(INFO) << "ROUND " << i;
        vector<vector<Tree>> parties_trees(parties.size());
        for (int p = 0; p < parties.size(); p++) {
            parties_trees[p].resize(params.gbdt_param.tree_per_rounds);
        }
//        vector<Tree> trees(params.gbdt_param.tree_per_rounds);

        GHPair sum_gh;
        for (int pid = 0; pid < parties.size(); pid++) {
            parties[pid].booster.update_gradients();
            GHPair party_gh = thrust::reduce(thrust::host, parties[pid].booster.gradients.host_data(), parties[pid].booster.gradients.host_end());
            sum_gh = sum_gh + party_gh;
        }

        LOG(INFO) << "SUM_GH" << sum_gh;

        // update gradients for all parties

//        SyncArray<GHPair> gh_pair(parties.size());
//        for(int i = 0; i < parties.size()-1; i++) {
//            parties[i].booster.update_gradients();
//            auto gh_pair_data = gh_pair.host_data();
//            for (int j = 0; j < parties[i].booster.gradients.size(); j++) {
//                gh_pair_data[i] = gh_pair_data[i] + parties[i].booster.gradients.host_data()[j];
//            }
//        }
//        LOG(INFO) << gh_pair;
//
//        GHPair sum_gh;
//        auto gh_pair_data = gh_pair.host_data();
//        for (int g = 0; g < gh_pair.size(); g++) {
//            sum_gh = sum_gh + gh_pair_data[g];
//        }

//        for (int j = 0; j < parties.size(); j++) {
//            LOG(INFO) << "Party update gradient";
//            parties[j].booster.update_gradients();
//            if (params.privacy_tech == "he") {
//                LOG(INFO) << "Encrypt gradient";
//                parties[i].booster.encrypt_gradients(parties[i].publicKey);
//            }else if (params.privacy_tech == "dp") {
//                LOG(INFO) << "Add DP noises to gradient";
//                parties[i].booster.add_noise_to_gradients(params.variance);
//            }
//        }
//
//        for (int i = 0; i < server.booster.gradients.size(); i++) {
//            auto gradient_data = server.booster.gradients.host_data();
//            if (std::isnan(gradient_data[i].g)) {
//                LOG(INFO) << "Gradient is nan";
//            }
//        }
//



        // for each tree per round
        for (int k = 0; k < params.gbdt_param.tree_per_rounds; k++) {
            LOG(INFO) << "CLASS" << k;
//            Tree &tree = trees[k];
            // each party initialize ins2node_id, gradients, etc.
            // ask parties to send gradient and aggregate by server
#pragma omp parallel for
            for (int pid = 0; pid < parties.size(); pid++) {
                parties[pid].booster.fbuilder->build_init(parties[pid].booster.gradients, k);
            }
            server.booster.fbuilder->build_init(sum_gh, k);


            // for each level
           // LOG(INFO) << "Level " << k;
            for (int d = 0; d < params.gbdt_param.depth; d++) {
               // LOG(INFO) << "Depth " << d;

                int n_nodes_in_level = 1 << d;
                int n_max_nodes = 2 << model_param.depth;
                MSyncArray<int> parties_hist_fid(parties.size());

                // Each Party Compute Histogram
                // each party compute hist, send hist to server or party


#pragma omp parallel for
                for (int j = 0; j < parties.size(); j++) {
                    int n_column = parties[j].dataset.n_features();
                    int n_partition = n_column * n_nodes_in_level;
                    int n_bins = parties[j].booster.fbuilder->cut.cut_points_val.size();
                    auto cut_fid_data = parties[j].booster.fbuilder->cut.cut_fid.host_data();
                    int n_max_splits = n_max_nodes * n_bins;

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
                    //todo: encrypt the histogram
                    aggregator.booster.fbuilder->append_hist(hist, missing_gh, n_partition, n_max_splits, j);
                }
                // Now we have the array of hist and missing_gh

                SyncArray <GHPair> missing_gh;
                SyncArray <GHPair> hist;
                int n_max_splits = n_max_nodes * n_bins;

                if (params.propose_split == "server") {
                    aggregator.booster.fbuilder->merge_histograms_server_propose(hist, missing_gh);
                    server.booster.fbuilder->set_last_hist(hist);
//                    LOG(INFO) << hist;
                }else if (params.propose_split == "client") {
                    // TODO: Fix this to make use of missing_gh
                    aggregator.booster.fbuilder->merge_histograms_client_propose(hist, missing_gh, n_max_splits);
                }

                // set these parameters to fit merged histogram

                n_bins = aggregator.booster.fbuilder->cut.cut_points_val.size();

                // server compute gain
                SyncArray <float_type> gain(n_max_splits * parties.size());

                // if privacy tech == 'he', decrypt histogram
//                if (params.privacy_tech == "he")
//                    server.decrypt_histogram();

                // if server propose cut, hist_fid for each party should be the same
                auto hist_fid_data = parties_hist_fid[0].host_data();
                server.booster.fbuilder->compute_gain_in_a_level(gain, n_nodes_in_level, n_bins, hist_fid_data,
                                                                 missing_gh, hist);
              //  LOG(INFO) << "GAIN:" << gain;
                // server find the best gain and its index
                SyncArray <int_float> best_idx_gain(n_nodes_in_level);
                server.booster.fbuilder->get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins);
//                LOG(INFO) << "BEST_IDX_GAIN:" << best_idx_gain;

                server.booster.fbuilder->get_split_points(best_idx_gain, n_nodes_in_level, hist_fid_data, missing_gh, hist);
               // LOG(INFO) << "SP" << server.booster.fbuilder->sp;
                server.booster.fbuilder->update_tree();

                // TODO: Update trees of every party
#pragma omp parallel for
                for (int j = 0; j < parties.size(); j++) {
                    parties_trees[j][k] = server.booster.fbuilder->get_tree();
                    parties[j].booster.fbuilder->set_tree(parties_trees[j][k]);
                    parties[j].booster.fbuilder->update_ins2node_id();
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

            // After training each tree, update vector of tree
#pragma omp parallel for
            for (int p = 0; p < parties.size(); p++) {
                Tree &tree = parties_trees[p][k];
                parties[p].booster.fbuilder->trees.prune_self(model_param.gamma);
                parties[p].booster.fbuilder->predict_in_training(k);
                tree.nodes.resize(parties[p].booster.fbuilder->trees.nodes.size());
                tree.nodes.copy_from(parties[p].booster.fbuilder->trees.nodes);
            }
        }
#pragma omp parallel for
        for (int p = 0; p < parties.size(); p++) {
            parties[p].gbdt.trees.push_back(parties_trees[p]);
        }
       LOG(INFO) <<  "Y_PREDICT" << parties[0].booster.fbuilder->get_y_predict()
                 << parties[0].booster.metric->get_name() << " = " << parties[0].booster.metric->get_score(parties[0].booster.fbuilder->get_y_predict());
//        std::ofstream myfile;
//        myfile.open ("data.txt", std::ios_base::app);
  //    myfile << parties[0].booster.metric->get_score(parties[0].booster.fbuilder->get_y_predict()) << "\n";
//        myfile.close();
    } LOG(INFO) << "end of training";
}

void FLtrainer::vertical_fl_trainer(vector<Party> &parties, Server &server, FLParam &params) {

    // load dataset
    GBDTParam &model_param = params.gbdt_param;
    Comm comm_helper;
    DifferentialPrivacy dp_manager;

    // initializing differential privacy
    if(params.privacy_tech == "dp") {
        dp_manager = DifferentialPrivacy();
        dp_manager.init(params);
    }

    // start training
    // for each boosting round

    std::chrono::high_resolution_clock timer;
    auto start = timer.now();

    for (int round = 0; round < params.gbdt_param.n_trees; round++) {

        vector<Tree> trees(params.gbdt_param.tree_per_rounds);

        // Server update, encrypt and send gradients
        server.booster.update_gradients();

        if (params.privacy_tech == "dp") {
            // option 1: direct add noise to gradients
//            server.booster.add_noise_to_gradients(params.variance);

            // option 2: clip gradients to (-1, 1)
            auto gradient_data = server.booster.gradients.host_data();
            for (int i = 0; i < server.booster.gradients.size(); i ++) {
//                LOG(INFO) << "before" << gradient_data[i].g;
                dp_manager.clip_gradient_value(gradient_data[i].g);
//                LOG(INFO) << "after" << gradient_data[i].g;
            }
        }

        SyncArray<GHPair> temp_gradients;
        if (params.privacy_tech == "he") {
            temp_gradients.resize(server.booster.gradients.size());
            temp_gradients.copy_from(server.booster.gradients);
            server.homo_init();
            server.encrypt_gh_pairs(server.booster.gradients);
        }

#pragma omp parallel for
        for (int j = 0; j < parties.size(); j++) {
            server.send_booster_gradients(parties[j]);
        }
        if (params.privacy_tech == "he") {
            server.booster.gradients.copy_from(temp_gradients);
        }

        // for each tree in a round
        for (int t = 0; t < params.gbdt_param.tree_per_rounds; t++) {
            Tree &tree = trees[t];
            // each party initialize ins2node_id, gradients, etc.
            server.booster.fbuilder->build_init(server.booster.gradients, t);

#pragma omp parallel for
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
                MSyncArray<int> parties_global_hist_fid(parties.size());
                MSyncArray<GHPair> parties_hist(parties.size());

                // each party compute hist, send hist to server
                for (int pid = 0; pid < parties.size(); pid++)
                    parties_n_columns[pid] = parties[pid].dataset.n_features();

#pragma omp parallel for
                for (int pid = 0; pid < parties.size(); pid++) {
                    int n_bins = parties[pid].booster.fbuilder->cut.cut_points_val.size();
                    parties_n_bins[pid] = n_bins;
                    int n_max_splits = n_max_nodes * n_bins;
                    int n_column = parties[pid].dataset.n_features();
                    int n_partition = n_column * n_nodes_in_level;
                    auto cut_fid_data = parties[pid].booster.fbuilder->cut.cut_fid.host_data();

                    SyncArray<int> hist_fid(n_nodes_in_level * n_bins);
                    SyncArray<int> global_hist_fid(n_nodes_in_level * n_bins);
                    auto hist_fid_data = hist_fid.host_data();
                    auto global_hist_fid_data = global_hist_fid.host_data();
                    int global_offset = accumulate(parties_n_columns.begin(), parties_n_columns.begin() + pid, 0);
                    for (int i = 0; i < hist_fid.size(); i++) {
                        hist_fid_data[i] = cut_fid_data[i % n_bins];
                        global_hist_fid_data[i] = hist_fid_data[i] + global_offset;
//                        global_hist_fid_data[i] = batch_idxs[pid][hist_fid_data[i]];
                    }
                    parties_hist_fid[pid].resize(n_nodes_in_level * n_bins);
                    parties_hist_fid[pid].copy_from(hist_fid);
                    parties_global_hist_fid[pid].resize(n_nodes_in_level * n_bins);
                    parties_global_hist_fid[pid].copy_from(global_hist_fid);

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

//                LOG(INFO) << parties_global_hist_fid[2];

                server.booster.fbuilder->sp.resize(n_nodes_in_level);
                // server concat hist_fid_data, missing_gh & histograms
                int n_bins_new = accumulate(parties_n_bins.begin(), parties_n_bins.end(), 0);
                int n_max_splits_new = n_max_nodes * n_bins_new;
                int n_column_new = accumulate(parties_n_columns.begin(), parties_n_columns.end(), 0);
                SyncArray<int> global_hist_fid(n_bins_new * n_nodes_in_level);
                SyncArray<GHPair> missing_gh(n_column_new * n_nodes_in_level);
                SyncArray<GHPair> hist(n_bins_new * n_nodes_in_level);
                global_hist_fid.copy_from(
                        comm_helper.concat_msyncarray(parties_global_hist_fid, parties_n_bins, n_nodes_in_level));
                missing_gh.copy_from(
                        comm_helper.concat_msyncarray(parties_missing_gh, parties_n_columns, n_nodes_in_level));
                hist.copy_from(comm_helper.concat_msyncarray(parties_hist, parties_n_bins, n_nodes_in_level));

                // server compute gain
                SyncArray<float_type> gain(n_max_splits_new);
                if (params.privacy_tech == "he") {
                    server.decrypt_gh_pairs(hist);
                    server.decrypt_gh_pairs(missing_gh);
                }
                server.booster.fbuilder->compute_gain_in_a_level(gain, n_nodes_in_level, n_bins_new,
                                                                 global_hist_fid.host_data(),
                                                                 missing_gh, hist, n_column_new);
//                LOG(INFO) << "gain:" << gain;
                for (int index = 0; index < gain.size(); index ++) {
                    if (gain.host_data()[index] != 0) {
//                        LOG(INFO) << gain.host_data()[index];
                    }
                }
                // server find the best gain and its index
                SyncArray<int_float> best_idx_gain(n_nodes_in_level);

                // with Exponential Mechanism: select with split probability
                if (params.privacy_tech == "dp") {
                    SyncArray<float_type> prob_exponent(n_max_splits_new);    //the exponent of probability mass for each split point
                    dp_manager.compute_split_point_probability(gain, prob_exponent);
                    auto prob_exponent_data = prob_exponent.host_data();
                    for (int index = 0; index < prob_exponent.size(); index ++) {
                        if(prob_exponent_data[index]!=0) {
                            LOG(INFO) << "prob expo: " << prob_exponent_data[index];
                        }
                    }
//                    LOG(INFO)<<prob_exponent;
                    dp_manager.exponential_select_split_point(prob_exponent, gain, best_idx_gain, n_nodes_in_level, n_bins_new);

                }
                // without Exponential Mechanism: select the split with max gain
                else {
                    server.booster.fbuilder->get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins_new);
                }
                LOG(INFO) << "best index gain: "<< best_idx_gain;
//                server.booster.fbuilder->get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins_new);

                auto best_idx_data = best_idx_gain.host_data();

                // parties who propose the best candidate update their trees accordingly
                vector<vector<int>> party_node_map(parties.size());
                for (int node = 0; node < n_nodes_in_level; node++) {

                    // convert the global best index to party id & its local index
                    int best_idx = get < 0 > (best_idx_data[node]);
                    best_idx -= node * n_bins_new;
                    float best_gain = get < 1 > (best_idx_data[node]);
                    int party_id = 0;
                    while (best_idx >= 0) {
                        best_idx -= parties_n_bins[party_id];
                        party_id += 1;
                    }
                    party_id -= 1;
                    int local_idx = best_idx + parties_n_bins[party_id] * (node + 1);

                    int node_shifted = node + (1 << l) - 1;
                    party_node_map[party_id].push_back(node_shifted);
                    // party get local split point
                    parties[party_id].booster.fbuilder->get_split_points_in_a_node(node, local_idx, best_gain,
                                                                                   n_nodes_in_level,
                                                                                   parties_hist_fid[party_id].host_data(),
                                                                                   parties_missing_gh[party_id],
                                                                                   parties_hist[party_id]);
                    // party update itself
                    parties[party_id].booster.fbuilder->update_tree_in_a_node(node);
                    parties[party_id].booster.fbuilder->update_ins2node_id_in_a_node(node_shifted);

                    // update local split_feature_id to global
                    auto party_global_hist_fid_data = parties_global_hist_fid[party_id].host_data();
                    int global_fid = party_global_hist_fid_data[local_idx];
                    auto nodes_data = parties[party_id].booster.fbuilder->trees.nodes.host_data();
                    auto sp_data = parties[party_id].booster.fbuilder->sp.host_data();
                    sp_data[node].split_fea_id = global_fid;
                    nodes_data[node_shifted].split_feature_id = global_fid;
                }

                // party broadcast new instance space to others
                vector<int> updated_parties;
                for (int sender_id = 0; sender_id < parties.size(); sender_id++) {
                    if (party_node_map[sender_id].size() > 0) {
                        updated_parties.push_back(sender_id);
                        for (int nid: party_node_map[sender_id]) {
                            parties[sender_id].send_node(nid, n_nodes_in_level, server);
                        }
                    }
                }

                if (params.privacy_tech == "he") {
                    auto node_data = server.booster.fbuilder->trees.nodes.host_data();
#pragma omp parallel for
                    for (int nid = (1 << l) - 1; nid < (2 << (l + 1)) - 1; nid++) {
                        server.decrypt_gh(node_data[nid].sum_gh_pair);
                        node_data[nid].calc_weight(params.gbdt_param.lambda);
                    }
                }

#pragma omp parallel for
                for (int pid = 0; pid < parties.size(); pid++) {
                    for (int nid = (1 << l) - 1; nid < (1 << l) - 1 + n_nodes_in_level; nid++) {
                        server.send_node(nid, n_nodes_in_level, parties[pid]);
                    }
                }

//                LOG(INFO) << "ins2node_id" << parties[0].booster.fbuilder->ins2node_id;
//                LOG(INFO) << parties[0].booster.fbuilder->trees.nodes;

                bool split_further = false;
                for (int pid:updated_parties) {
                    if (parties[pid].booster.fbuilder->has_split) {
                        split_further = true;
                        break;
                    }
                }
                if (!split_further) {
                    // add Laplace noise to leaf node values
                    if (params.privacy_tech == "dp") {
                        for(int party_id = 0; party_id < parties.size(); party_id ++) {
                            int tree_size = parties[party_id].booster.fbuilder->trees.nodes.size();
                            auto nodes = parties[party_id].booster.fbuilder->trees.nodes.host_data();
                            for(int node_id = 0; node_id < tree_size; node_id++) {
                                Tree::TreeNode node = nodes[node_id];
                                if(node.is_leaf) {
                                    // add noises
                                    dp_manager.laplace_add_noise(node);
                                }
                            }
                        }
                    }
                    break;
                }
            }

            for (int pid = 0; pid < parties.size(); pid++) {
                parties[pid].booster.fbuilder->trees.prune_self(model_param.gamma);
                parties[pid].booster.fbuilder->predict_in_training(t);
            }
            server.booster.fbuilder->trees.prune_self(model_param.gamma);
            server.booster.fbuilder->predict_in_training(t);
            tree.nodes.resize(parties[0].booster.fbuilder->trees.nodes.size());
            tree.nodes.copy_from(parties[0].booster.fbuilder->trees.nodes);
        }

//        LOG(INFO) << "y_predict: " << parties[0].booster.fbuilder->get_y_predict();
        parties[0].gbdt.trees.push_back(trees);
        LOG(INFO) << parties[0].booster.metric->get_name() << " = "
                  << parties[0].booster.metric->get_score(parties[0].booster.fbuilder->get_y_predict());
    }

    auto stop = timer.now();
    std::chrono::duration<float> training_time = stop - start;
    LOG(INFO) << "training time = " << training_time.count();

    LOG(INFO) << "end of training";
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