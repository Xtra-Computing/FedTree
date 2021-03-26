//
// Created by liqinbin on 10/14/20.
//
#include "FedTree/FL/FLtrainer.h"
#include "FedTree/FL/partition.h"
#include "FedTree/FL/comm_helper.h"
#include "thrust/sequence.h"

using namespace thrust;

void FLtrainer::horizontal_fl_trainer(vector<Party> &parties, Server &server, FLParam &params) {
    /*
    // load dataset
    GBDTParam &model_param = params.gbdt_param;
    DataSet dataset;
    dataset.load_from_file(model_param.path, params);

    int parties_size = params.n_parties;
    // partition dataset
    vector<DataSet> subsets(parties_size);
    Partition partition;
    vector<float> alpha;
    if (params.partition) {
        partition.hetero_partition(dataset, parties_size, true, subsets, alpha);
    }
    vector<int> n_instances_per_party = vector<int>();
    for(const DataSet& ds: subsets) {
        n_instances_per_party.push_back(ds.n_instances());
    }

    // server and party initialization
    server.init(params, dataset.n_instances(), n_instances_per_party);
    if (params.privacy_tech == "he") {
        server.homo_init();
    }
    for (int i = 1; i < parties_size; i++) {
        parties[i - 1].init(i, subsets[i], params);
    }

    vector<vector<Tree>> trees(params.gbdt_param.n_trees);
    for (int i = 0; i < params.gbdt_param.n_trees; i++) {
        for (int j = 0; j < parties_size; j++) {
            parties[j].booster.update_gradients();
        }
        vector<Tree> trees_per_round(params.gbdt_param.tree_per_rounds);
        for(int l = 0; l < params.gbdt_param.tree_per_rounds; l++) {
            Tree &tree = trees_per_round[l];
            Tree* partial_tree;
            for (int j = 0; j < params.gbdt_param.depth; j++) {
                // send histograms from party to server
                for (int k = 1; k < parties.size(); k++) {
                    parties[k].send_gradients(server);
                }
                partial_tree = server.booster.fbuilder->build_tree_level_approximate(j, l);
                // NULL pointer indicates that tree nodes cannot be further split.
                if(!partial_tree) {
                    break;
                }
                for(int k = 1; k < parties.size(); k ++) {
                    server.send_trees(parties[k]);
                }
            }
            server.booster.fbuilder->get_tree().prune_self(params.gbdt_param.gamma);
            server.booster.fbuilder->set_y_predict(l);
            tree.nodes.resize(server.booster.fbuilder->get_tree().nodes.size());
            tree.nodes.copy_from(server.booster.fbuilder->get_tree().nodes);
        }
//          server.booster.boost(trees);
        trees.push_back(trees_per_round);
    }
    for (int j = 0; j < parties_size; j++) {
        parties[j].gbdt.trees = trees;
    }
     */
}

//for (int j = 0; j < params.gbdt_param.depth; j++) {
//for (int k = 0; k < parties.size(); k++) {
//parties[j].send_gradients(server);
//}
//}


//void FLtrainer::horizontal_fl_trainer(vector<Party> &parties, Server &server, FLParam &params) {
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

void FLtrainer::vertical_fl_trainer(vector<Party> &parties, Server &server, FLParam &params) {

    // load dataset
    GBDTParam &model_param = params.gbdt_param;
    Comm comm_helper;

    // start training
    // for each boosting round

    std::chrono::high_resolution_clock timer;
    auto start = timer.now();

    for (int round = 0; round < params.gbdt_param.n_trees; round++) {

        vector<Tree> trees(params.gbdt_param.tree_per_rounds);

        // Server update, encrypt and send gradients
        server.booster.update_gradients();

        SyncArray<GHPair> temp_gradients;
        if (params.privacy_tech == "he") {
            temp_gradients.resize(server.booster.gradients.size());
            temp_gradients.copy_from(server.booster.gradients);
            server.homo_init();
//            server.booster.encrypt_gradients(server.publicKey);
            server.encrypt_gh_pairs(server.booster.gradients);
        }
        if (params.privacy_tech == "dp");
//            server.booster.add_noise_to_gradients(params.variance);

#pragma omp parallel for
        for (int j = 0; j < parties.size(); j++) {
            server.send_booster_gradients(parties[j]);
        }
        if (params.privacy_tech == "he") {
            server.booster.gradients.copy_from(temp_gradients);
        }

//        server.booster.update_gradients();
//        for (int pid = 0; pid < parties.size(); pid++)
//            parties[pid].booster.update_gradients();

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
//                std::ofstream myfile;
//                myfile.open(std::to_string(l) + ".txt");
//                auto gain_data = gain.host_data();
//                for (int tmp = 0; tmp < gain.size(); tmp++) {
//                    myfile << gain_data[tmp] << '\t';
//                }
//                myfile.close();

                // server find the best gain and its index
                SyncArray<int_float> best_idx_gain(n_nodes_in_level);
                server.booster.fbuilder->get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins_new);
//                LOG(INFO) << "best_idx_gain:" << best_idx_gain;
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
                    for (int nid = (2 << l) - 1; nid < (2 << (l + 1)) - 1; nid++) {
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

                bool split_further = false;
                for (int pid:updated_parties) {
                    if (parties[pid].booster.fbuilder->has_split) {
                        split_further = true;
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