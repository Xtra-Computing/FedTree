// Created by liqinbin on 10/14/20.
//
#include "FedTree/DP/differential_privacy.h"
#include "FedTree/FL/FLtrainer.h"
#include "FedTree/FL/partition.h"
#include "FedTree/FL/comm_helper.h"
#include "thrust/sequence.h"
#include "thrust/reduce.h"
#include "thrust/execution_policy.h"
#include <limits>
#include <cmath>

using namespace thrust;
void FLtrainer::horizontal_fl_trainer(vector<Party> &parties, Server &server, FLParam &params) {
    LOG(INFO) << "Start horizontal training";
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();
    auto start = t_start;
    auto model_param = params.gbdt_param;
    Party &aggregator = (params.merge_histogram == "client")? parties[0] : server;
    int n_parties = parties.size();
    aggregator.booster.fbuilder->parties_hist_init(n_parties);

    std::vector<float> encryption_time (n_parties, 0.0f);
    float decryption_time = 0.0f;
    if (params.privacy_tech == "he") {
        LOG(INFO) << "Start HE Init";
        // server generate public key and private key
        server.homo_init();
        // server distribute public key to rest of parties
        for (int i = 0; i < n_parties; i++) {
            parties[i].paillier = server.paillier;
        }
        LOG(INFO) << "End of HE init";
    }
    else if(params.privacy_tech == "sa"){
        LOG(INFO)<<"Start SA init";
//        server.dh.primegen();
        for(int i = 0; i < n_parties; i++){
//            parties[i].dh = server.dh;
            parties[i].dh.generate_public_key();
        }
        for(int i = 0; i < n_parties; i++){
            parties[i].dh.pid = i;
            parties[i].dh.init_variables(n_parties);
            for(int j = 0; j < n_parties; j++) {
                if(j != i) {
                    parties[i].dh.other_public_keys[j] = parties[j].dh.public_key;
                }
            }
            parties[i].dh.compute_shared_keys();
        }
        LOG(INFO)<<"End of SA init";
    }
    DifferentialPrivacy dp_manager = DifferentialPrivacy();
    if (params.privacy_tech == "dp"){
        LOG(INFO) << "Start DP init";
        dp_manager.init(params);
        LOG(INFO) << "End of DP init";
    }
    
    // Generate HistCut by server or each party
    int n_bins = model_param.max_num_bin;
    if (params.propose_split == "server") {
        // loop through all party to find max and min for each feature
        float inf = std::numeric_limits<float>::infinity();
        vector<vector<float>> feature_range(parties[0].get_num_feature());
        for (int n = 0; n < parties[0].get_num_feature(); n++) {
            vector<float> min_max = {inf, -inf};
            for (int p = 0; p < n_parties; p++) {
                vector<float> temp = parties[p].get_feature_range_by_feature_index(n);
                if (temp[0] <= min_max[0])
                    min_max[0] = temp[0];
                if (temp[1] >= min_max[1])
                    min_max[1] = temp[1];
            }
            feature_range[n] = min_max;
        }
        // once we have feature_range, we can generate cut points
        server.booster.fbuilder->cut.get_cut_points_by_feature_range(feature_range, n_bins);

        for (int p = 0; p < n_parties; p++) {
            parties[p].booster.fbuilder->set_cut(server.booster.fbuilder->cut);
            parties[p].booster.fbuilder->get_bin_ids();
        }

    } else if (params.propose_split == "client" || params.propose_split == "client_pre" || params.propose_split == "client_post") {
        for (int p = 0; p < n_parties; p++) {
            auto dataset = parties[p].dataset;
            parties[p].booster.fbuilder->cut.get_cut_points_fast(dataset, n_bins, dataset.n_instances());
            aggregator.booster.fbuilder->append_to_parties_cut(parties[p].booster.fbuilder->cut, p);
        }
        if (params.propose_split == "client_pre") {
            // find feature range of each feature for each party
            int n_columns = aggregator.booster.fbuilder->parties_cut[0].cut_col_ptr.size() - 1;
            vector<vector<float>> ranges(n_columns);

            // Merging all cut points into one single cut points
            for (int n = 0; n < n_columns; n++) {
                for (int p = 0; p < aggregator.booster.fbuilder->parties_cut.size(); p++) {
                    auto parties_cut_col_data = aggregator.booster.fbuilder->parties_cut[p].cut_col_ptr.host_data();
                    auto parties_cut_points_val_data = aggregator.booster.fbuilder->parties_cut[p].cut_points_val.host_data();

                    int column_start = parties_cut_col_data[n];
                    int column_end = parties_cut_col_data[n+1];

                    for (int i = column_start; i < column_end; i++) {
                        ranges[n].push_back(parties_cut_points_val_data[i]);
                    }
                }
            }

            // Once we have gathered the sorted range, we can randomly sample the cut points to match with the number of bins
            int n_features = ranges.size();
            int max_num_bins = aggregator.booster.fbuilder->parties_cut[0].cut_points_val.size() / n_columns + 1;
            // The vales of cut points
            SyncArray<float_type> cut_points_val;
            // The number of accumulated cut points for current feature
            SyncArray<int> cut_col_ptr;
            // The feature id for current cut point
            SyncArray<int> cut_fid;
            cut_points_val.resize(n_features * max_num_bins);
            cut_col_ptr.resize(n_features + 1);
            cut_fid.resize(n_features * max_num_bins);
            auto cut_points_val_data = cut_points_val.host_data();
            auto cut_col_ptr_data = cut_col_ptr.host_data();
            auto cut_fid_data = cut_fid.host_data();

            int index = 0;
            for (int fid = 0; fid < n_features; fid++) {
                vector<float> sample;
                cut_col_ptr_data[fid] = index;

                // Always keep the maximum value
                if (ranges[fid].size() > 0) {
                    auto max_element = *std::max_element(ranges[fid].begin(), ranges[fid].end());
                    sample.push_back(max_element);
                } else continue;

                // Randomly sample number of cut point according to max num bins
                unsigned seed = chrono::steady_clock::now().time_since_epoch().count();
                std::shuffle(ranges[fid].begin(), ranges[fid].end(), std::default_random_engine(seed));

                struct compare
                {
                    int key;
                    compare(int const &i): key(i) {}
                    bool operator()(int const &i) {
                        return (i == key);
                    }
                };

                for (int i = 0; i < ranges[fid].size(); i++) {
                    if (sample.size() == max_num_bins)
                        break;
                    auto element = ranges[fid][i];
                    // Check if element already in cut points val data
                    if (not (std::find(sample.begin(), sample.end(), element) != sample.end()))
                        sample.push_back(element);
                }

                // Sort the sample in descending order
                std::sort(sample.begin(), sample.end(), std::greater<float>());

                // Populate cut points val with samples
                for (int i = 0; i < sample.size(); i++) {
                    cut_points_val_data[index] = sample[i];
                    cut_fid_data[index] = fid;
                    index++;
                }
            }
            cut_col_ptr_data[n_features] = index;

            HistCut cut;
            cut.cut_points_val.resize(cut_points_val.size());
            cut.cut_points_val.copy_from(cut_points_val);
            cut.cut_fid.resize(cut_fid.size());
            cut.cut_fid.copy_from(cut_fid);
            cut.cut_col_ptr.resize(cut_col_ptr.size());
            cut.cut_col_ptr.copy_from(cut_col_ptr);
            server.booster.fbuilder->set_cut(cut);

            // Distribute cut points to all parties

            for (int p = 0; p < parties.size(); p++) {
                parties[p].booster.fbuilder->set_cut(server.booster.fbuilder->cut);
                parties[p].booster.fbuilder->get_bin_ids();
            }
        }
    }
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(DEBUG) << "Initialization using time: " << used_time.count() << " s";
    t_start = t_end;

    for (int i = 0; i < params.gbdt_param.n_trees; i++) {
        LOG(INFO) << "Training round " << i << " start";
        vector<Tree> trees_this_round;
        trees_this_round.resize(params.gbdt_param.tree_per_rounds);
//        vector<Tree> trees(params.gbdt_param.tree_per_rounds);
        // each party sample the data to train a tree in each round
        if(params.ins_bagging_fraction < 1.0){
            if(i % int(1/params.ins_bagging_fraction) == 0){
                for(int i = 0; i < n_parties; i++){
                    parties[i].bagging_init();
                }
            }
            for(int pid = 0; pid<n_parties; pid++) {
                parties[pid].sample_data();
                parties[pid].booster.init(parties[pid].dataset, params.gbdt_param);
                parties[pid].booster.fbuilder->set_cut(server.booster.fbuilder->cut);
                parties[pid].booster.fbuilder->get_bin_ids();
//                SyncArray<float_type> y_predict = parties[pid].booster.fbuilder->get_y_predict();
                //reset y_predict
                if(i!=0)
                    parties[pid].gbdt.predict_raw(params.gbdt_param, parties[pid].dataset,
                                                parties[pid].booster.fbuilder->get_y_predict());
            }
        }
        GHPair sum_gh;
        for (int pid = 0; pid < n_parties; pid++) {
            parties[pid].booster.update_gradients();
            if(params.privacy_tech == "dp"){
                auto gradient_data = parties[pid].booster.gradients.host_data();
                for(int i = 0; i < parties[pid].booster.gradients.size(); i++){
                    dp_manager.clip_gradient_value(gradient_data[i].g);
                }
            }

            GHPair party_gh = thrust::reduce(thrust::host, parties[pid].booster.gradients.host_data(), parties[pid].booster.gradients.host_end());
            sum_gh = sum_gh + party_gh;
        }
        // for each tree per round
        for (int k = 0; k < params.gbdt_param.tree_per_rounds; k++) {
//            Tree &tree = trees[k];
            // each party initialize ins2node_id, gradients, etc.
            // ask parties to send gradient and aggregate by server
            #pragma omp parallel for
            for (int pid = 0; pid < n_parties; pid++) {
                parties[pid].booster.fbuilder->build_init(parties[pid].booster.gradients, k);
            }
            server.booster.fbuilder->build_init(sum_gh, k);

            t_end = timer.now();
            used_time = t_end - t_start;
            LOG(DEBUG) << "Initializing builder using time: " << used_time.count() << " s";
            t_start = t_end;
            // for each level
           // LOG(INFO) << "Level " << k;
            for (int d = 0; d < params.gbdt_param.depth; d++) {
                int n_nodes_in_level = 1 << d;
                int n_max_nodes = 2 << model_param.depth;
                MSyncArray<int> parties_hist_fid(n_parties);

                // Each Party Compute Histogram
                // each party compute hist, send hist to server or party
                if (params.privacy_tech == "sa"){
                    for(int i = 0; i < n_parties; i++) {
                        parties[i].dh.generate_noises();
                    }
                    for(int i = 0; i < n_parties; i++) {
                        for(int j = 0; j < n_parties; j++) {
                            if(j!=i)
                                parties[i].dh.received_encrypted_noises[j] = parties[j].dh.encrypted_noises[i];
                        }
                        parties[i].dh.decrypt_noises();
                    }
                }
                #pragma omp parallel for
                for (int j = 0; j < n_parties; j++) {
                    int n_column = parties[j].dataset.n_features();
                    int n_partition = n_column * n_nodes_in_level;
                    int n_bins = parties[j].booster.fbuilder->cut.cut_points_val.size();
                    auto cut_fid_data = parties[j].booster.fbuilder->cut.cut_fid.host_data();
                    int n_max_splits = n_max_nodes * n_bins;
                    int n_splits = n_nodes_in_level * n_bins;
                    SyncArray<int> hist_fid(n_splits);
                    auto hist_fid_data = hist_fid.host_data();

                    for (int i = 0; i < hist_fid.size(); i++)
                        hist_fid_data[i] = cut_fid_data[i % n_bins];

                    parties_hist_fid[j].resize(n_nodes_in_level * n_bins);
                    parties_hist_fid[j].copy_from(hist_fid);

                    SyncArray <GHPair> missing_gh(n_partition);
                    SyncArray <GHPair> hist(n_splits);
                    // FIXME n_max_splits has no effect in this function
                    parties[j].booster.fbuilder->compute_histogram_in_a_level(d, n_max_splits, n_bins,
                                                                              n_nodes_in_level,
                                                                              hist_fid_data, missing_gh, hist);
                    t_end = timer.now();
                    used_time = t_end - t_start;
                    LOG(DEBUG) << "Computing histogram using time: " << used_time.count() << " s";
                    t_start = t_end;
                    // encrypt the histogram
                    if (params.privacy_tech == "he") {
                        auto t1 = timer.now();
                        parties[j].encrypt_histogram(hist);
                        parties[j].encrypt_histogram(missing_gh);
                        auto t2 = timer.now();
                        std::chrono::duration<float> t3 = t2 - t1;
                        encryption_time[j] += t3.count();
                    }
                    else if(params.privacy_tech == "sa"){
                        parties[j].add_noise_to_histogram(hist);
                        parties[j].add_noise_to_histogram(missing_gh);
                    }

                    aggregator.booster.fbuilder->append_hist(hist, missing_gh, n_partition, n_splits, j);

                    t_end = timer.now();
                    used_time = t_end - t_start;
                    LOG(DEBUG) << "Appending histogram using time: " << used_time.count() << " s";
                    t_start = t_end;
                }
                // Now we have the array of hist and missing_gh


                SyncArray <GHPair> missing_gh;
                SyncArray <GHPair> hist;
                // set these parameters to fit merged histogram
                n_bins = aggregator.booster.fbuilder->cut.cut_points_val.size();
                int n_max_splits = n_max_nodes * n_bins;
                if (params.propose_split == "server" || params.propose_split == "client_pre") {
//                    if (params.privacy_tech == "he") {
//                        hist.resize(aggregator.booster.fbuilder->parties_hist[0].size());
//                        missing_gh.resize(aggregator.booster.fbuilder->parties_missing_gh[0].size());
//                        aggregator.encrypt_histogram(hist);
//                        aggregator.encrypt_histogram(missing_gh);
//
//                    }
                    aggregator.booster.fbuilder->merge_histograms_server_propose(hist, missing_gh);
//                    server.booster.fbuilder->set_last_missing_gh(missing_gh);
//                    LOG(INFO) << hist;
                }else if (params.propose_split == "client_post") {
                    vector<vector<vector<float>>> feature_range_for_client(parties[0].get_num_feature());
                    for (int n = 0; n < parties[0].get_num_feature(); n++) {
                        feature_range_for_client[n].resize(parties.size());
                        for (int p = 0; p < parties.size(); p++) {
                            vector<float> temp = parties[p].get_feature_range_by_feature_index(n);
                            feature_range_for_client[n][p] = temp;
                        }
                    }
//                    if(params.privacy_tech == "he"){
//                        hist.resize(n_max_splits);
//                        missing_gh.resize(aggregator.booster.fbuilder->parties_missing_gh[0].size());
//                        aggregator.encrypt_histogram(hist);
//                        aggregator.encrypt_histogram(missing_gh);
//                    }
                    aggregator.booster.fbuilder->merge_histograms_client_propose(hist, missing_gh, feature_range_for_client, n_max_splits);
                }
                n_bins = aggregator.booster.fbuilder->cut.cut_points_val.size();

                // server compute gain
                SyncArray <float_type> gain(n_max_splits);

                // if privacy tech == 'he', decrypt histogram
                if (params.privacy_tech == "he") {
                    auto t1 = timer.now();
                    server.decrypt_gh_pairs(hist);
                    server.decrypt_gh_pairs(missing_gh);
                    auto t2 = timer.now();
                    std::chrono::duration<float> t3 = t2 - t1;
                    decryption_time += t3.count();
                }
                // if server propose cut, hist_fid for each party should be the same
                auto hist_fid_data = parties_hist_fid[0].host_data();
                server.booster.fbuilder->compute_gain_in_a_level(gain, n_nodes_in_level, n_bins, hist_fid_data,
                                                                 missing_gh, hist);

                // server find the best gain and its index
                SyncArray <int_float> best_idx_gain(n_nodes_in_level);
                if(params.privacy_tech == "dp"){
//                    SyncArray<float_type> prob_exponent(n_max_splits);    //the exponent of probability mass for each split point
//                    dp_manager.compute_split_point_probability(gain, prob_exponent);
//                    dp_manager.exponential_select_split_point(prob_exponent, gain, best_idx_gain, n_nodes_in_level, n_bins);

                    server.booster.fbuilder->get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins);
                }
                else
                    server.booster.fbuilder->get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins);
                LOG(DEBUG) << "BEST_IDX_GAIN:" << best_idx_gain;

                server.booster.fbuilder->get_split_points(best_idx_gain, n_nodes_in_level, hist_fid_data, missing_gh, hist);
                LOG(DEBUG) << "SP" << server.booster.fbuilder->sp;
                server.booster.fbuilder->update_tree();

                trees_this_round[k] = server.booster.fbuilder->get_tree();
                #pragma omp parallel for
                for (int j = 0; j < n_parties; j++) {
//                    parties_trees[j][k] = server.booster.fbuilder->get_tree();
//                    parties[j].booster.fbuilder->set_tree(parties_trees[j][k]);
                    parties[j].booster.fbuilder->set_tree(trees_this_round[k]);
                    parties[j].booster.fbuilder->update_ins2node_id();
                }

                bool split_further = true;
                for (int pid = 0; pid < n_parties; pid++) {
                    if (!parties[pid].booster.fbuilder->has_split) {
                        split_further = false;
                        break;
                    }
                }
                if (!split_further) {
//                    if (params.privacy_tech == "dp") {
//                        for (int party_id = 0; party_id < parties.size(); party_id++) {
//                            int tree_size = parties[party_id].booster.fbuilder->trees.nodes.size();
//                            auto nodes = parties[party_id].booster.fbuilder->trees.nodes.host_data();
//                            for (int node_id = 0; node_id < tree_size; node_id++) {
//                                Tree::TreeNode node = nodes[node_id];
//                                if (node.is_leaf) {
//                                    // add noises
//                                    dp_manager.laplace_add_noise(node);
//                                }
//                            }
//                        }
//                    }
                    break;
                }
            }

            t_end = timer.now();
            used_time = t_end - t_start;
            LOG(DEBUG) << "Building tree using time: " << used_time.count() << " s";
            t_start = t_end;

            // After training each tree, update vector of tree
            server.booster.fbuilder->trees.prune_self(model_param.gamma);
            Tree &tree = trees_this_round[k];
            #pragma omp parallel for
            for (int p = 0; p < n_parties; p++) {
                parties[p].booster.fbuilder->trees.prune_self(model_param.gamma);
                parties[p].booster.fbuilder->predict_in_training(k);
            }
            tree.nodes.resize(parties[0].booster.fbuilder->trees.nodes.size());
            tree.nodes.copy_from(parties[0].booster.fbuilder->trees.nodes);

            t_end = timer.now();
            used_time = t_end - t_start;
            LOG(DEBUG) << "Pruning tree using time: " << used_time.count() << " s";
            t_start = t_end;
        }
        #pragma omp parallel for
        for (int p = 0; p < n_parties; p++) {
//            parties[p].gbdt.trees.push_back(parties_trees[p]);
            parties[p].gbdt.trees.push_back(trees_this_round);
        }
        server.global_trees.trees.push_back(trees_this_round);
       // LOG(INFO) <<  "Y_PREDICT" << parties[0].booster.fbuilder->get_y_predict();
       float score = 0.0;
       for (int p = 0; p < n_parties; p++){
           score += parties[p].booster.metric->get_score(parties[p].booster.fbuilder->get_y_predict());
       }
       score /= n_parties;

       LOG(INFO) << "averaged " << parties[0].booster.metric->get_name() << " = "
                  << score;
        LOG(INFO) << "Training round " << i << " end";
    }
    auto t_stop = timer.now();
    std::chrono::duration<float> training_time = t_stop - start;
    LOG(INFO) << "training time = " << training_time.count() << "s";
    if (params.privacy_tech == "he"){
        for(int i = 0; i < n_parties; i++){
            LOG(INFO) << "party " << i << " HE time (encryption and decryption) " << encryption_time[i] + decryption_time << "("
                << encryption_time[i] << "/" << decryption_time << ")";
        }
        LOG(INFO) << "avg HE time " << std::accumulate(encryption_time.begin(), encryption_time.end(), 0.0)/n_parties + decryption_time;
    }

    LOG(INFO) << "end of training";
}

void FLtrainer::vertical_fl_trainer(vector<Party> &parties, Server &server, FLParam &params) {
    std::chrono::high_resolution_clock timer;
    auto start = timer.now();
    float encryption_time = 0.0f;
    float decryption_time = 0.0f;
    // load dataset
    GBDTParam &model_param = params.gbdt_param;
    Comm comm_helper;
    DifferentialPrivacy dp_manager;
    int n_parties = parties.size();
    // initializing differential privacy
    if (params.privacy_tech == "dp") {
        dp_manager = DifferentialPrivacy();
        dp_manager.init(params);
    }
//    if(params.ins_bagging_fraction < 1.0){
//        LOG(INFO)<<"start bagging init";
//        for(int i = 0; i < n_parties; i++){
//            parties[i].bagging_init(36);
//        }
//        server.bagging_init(36);
//    }
    // start training
    // for each boosting round

    for (int round = 0; round < params.gbdt_param.n_trees; round++) {
        LOG(INFO) << "Training round " << round << " start";

        vector<Tree> trees(params.gbdt_param.tree_per_rounds);

        if(params.ins_bagging_fraction < 1.0){
            if(round % (int(1/params.ins_bagging_fraction)) == 0) {
                for(int i = 0; i < n_parties; i++){
                    parties[i].bagging_init(36);
                }
                server.bagging_init(36);
            }
            server.sample_data();
            server.booster.init(server.dataset, params.gbdt_param);
            if(round!=0){
                server.predict_raw_vertical_jointly_in_training(params.gbdt_param, parties,
                                                                server.booster.fbuilder->get_y_predict());
            }
            for(int pid = 0; pid < n_parties; pid++) {
                parties[pid].sample_data();
                parties[pid].booster.init(parties[pid].dataset, params.gbdt_param);
            }
        }
        // Server update, encrypt and send gradients
        server.booster.update_gradients();

        if (params.privacy_tech == "dp") {
            // option 1: direct add noise to gradients
//            server.booster.add_noise_to_gradients(params.variance);

            // option 2: clip gradients to (-1, 1)
            auto gradient_data = server.booster.gradients.host_data();
            for (int i = 0; i < server.booster.gradients.size(); i++) {
                dp_manager.clip_gradient_value(gradient_data[i].g);
            }
        }
        // temp_gradients store the raw gradients
        SyncArray<GHPair> temp_gradients;
        if (params.privacy_tech == "he") {
            auto t1 = timer.now();
            temp_gradients.resize(server.booster.gradients.size());
            temp_gradients.copy_from(server.booster.gradients);
            server.homo_init();
            server.encrypt_gh_pairs(server.booster.gradients);
            auto t2 = timer.now();
            std::chrono::duration<float> t3 = t2 - t1;
            encryption_time += t3.count();
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
                bool split_further=false;
                int n_nodes_in_level = 1 << l;
                int n_max_nodes = 2 << model_param.depth;
                vector<int> parties_n_bins(parties.size());
                vector<int> parties_n_columns(parties.size());
                MSyncArray<GHPair> parties_missing_gh(parties.size());
                MSyncArray<int> parties_hist_fid(parties.size());
                MSyncArray<int> parties_global_hist_fid(parties.size());
                MSyncArray<GHPair> parties_hist(parties.size());

                // each party computes hist, sends hist to server
                for (int pid = 0; pid < parties.size(); pid++)
                    parties_n_columns[pid] = parties[pid].dataset.n_features();

                #pragma omp parallel for
                for (int pid = 0; pid < parties.size(); pid++) {
                    int n_bins = parties[pid].booster.fbuilder->cut.cut_points_val.size();
//                    std::cout<<"party "<<pid<<" n_bins:"<<n_bins<<std::endl;
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
                    SyncArray<GHPair> hist(n_nodes_in_level * n_bins);
                    parties[pid].booster.fbuilder->compute_histogram_in_a_level(l, n_max_splits, n_bins,
                                                                                n_nodes_in_level,
                                                                                hist_fid_data, missing_gh, hist);

                    parties_missing_gh[pid].resize(n_partition);
                    parties_missing_gh[pid].copy_from(missing_gh);
                    parties_hist[pid].resize(n_bins * n_nodes_in_level);
                    parties_hist[pid].copy_from(hist);

                    parties[pid].booster.fbuilder->sp.resize(n_nodes_in_level);
                }

                server.booster.fbuilder->sp.resize(n_nodes_in_level);
                // server concat hist_fid_data, missing_gh & histograms
                int n_bins_new = accumulate(parties_n_bins.begin(), parties_n_bins.end(), 0);
                int n_max_splits_new = n_max_nodes * n_bins_new;
                int n_column_new = accumulate(parties_n_columns.begin(), parties_n_columns.end(), 0);
                SyncArray<int> global_hist_fid(n_bins_new * n_nodes_in_level);
                SyncArray<GHPair> missing_gh(n_column_new * n_nodes_in_level);
                SyncArray<GHPair> hist(n_bins_new * n_nodes_in_level);
                global_hist_fid.copy_from(
                        comm_helper.concat_msyncarray(parties_global_hist_fid, n_nodes_in_level));
                missing_gh.copy_from(
                        comm_helper.concat_msyncarray(parties_missing_gh, n_nodes_in_level));
                hist.copy_from(comm_helper.concat_msyncarray(parties_hist, n_nodes_in_level));

                // server compute gain
                SyncArray<float_type> gain(n_max_splits_new);
                if (params.privacy_tech == "he") {
                    auto t1 = timer.now();
                    server.decrypt_gh_pairs(hist);
                    server.decrypt_gh_pairs(missing_gh);
                    auto t2 = timer.now();
                    std::chrono::duration<float> t3 = t2 - t1;
                    decryption_time += t3.count();
                }
                // LOG(INFO) << "hist:"<<"\n"<<hist;
                server.booster.fbuilder->compute_gain_in_a_level(gain, n_nodes_in_level, n_bins_new,
                                                                 global_hist_fid.host_data(),
                                                                 missing_gh, hist);
                // server find the best gain and its index
                SyncArray<int_float> best_idx_gain(n_nodes_in_level);

                // with Exponential Mechanism: select with split probability
                if (params.privacy_tech == "dp") {
                    SyncArray<float_type> prob_exponent(
                            n_max_splits_new);    //the exponent of probability mass for each split point
                    dp_manager.compute_split_point_probability(gain, prob_exponent);
                    auto prob_exponent_data = prob_exponent.host_data();
                    dp_manager.exponential_select_split_point(prob_exponent, gain, best_idx_gain, n_nodes_in_level,
                                                              n_bins_new);
                }
                    // without Exponential Mechanism: select the split with max gain
                else {
                    server.booster.fbuilder->get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level,
                                                                      n_bins_new);
                }
                LOG(DEBUG) << "best_idx_gain:" << best_idx_gain;
                auto best_idx_data = best_idx_gain.host_data();

                // parties who propose the best candidate update their trees accordingly
                vector<vector<int>> party_node_map(parties.size());

                for (int node = 0; node < n_nodes_in_level; node++) {
                    auto server_nodes_data = server.booster.fbuilder->trees.nodes.host_data();
                    if (!server_nodes_data[node + n_nodes_in_level - 1].is_valid) {
                        continue;
                    }
                    // convert the global best index to party id & its local index
                    int best_idx = get<0>(best_idx_data[node]);
                    best_idx -= node * n_bins_new;
                    float best_gain = get<1>(best_idx_data[node]);
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

                    // LOG(INFO) << "sp:"<<"\n"<<parties[party_id].booster.fbuilder->sp;
                    // party update itself
                    parties[party_id].booster.fbuilder->update_tree_in_a_node(node);
                    parties[party_id].booster.fbuilder->update_ins2node_id_in_a_node(node_shifted);

                    // LOG(INFO)<<"level "<<l<<":tree nodes"<<"\n"<<parties[party_id].booster.fbuilder->trees.nodes;

                    if (!split_further) {
                        if (parties[party_id].booster.fbuilder->has_split) {
                            split_further = true;
                        }
                    }
                    // update local split_feature_id to global
                    auto party_global_hist_fid_data = parties_global_hist_fid[party_id].host_data();
                    int global_fid = party_global_hist_fid_data[local_idx];
                    auto nodes_data = parties[party_id].booster.fbuilder->trees.nodes.host_data();
                    auto sp_data = parties[party_id].booster.fbuilder->sp.host_data();
                    // LOG(INFO)<<"local split fea id:"<<sp_data[node].split_fea_id;
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
                    auto t1 = timer.now();
                    auto node_data = server.booster.fbuilder->trees.nodes.host_data();

                    #pragma omp parallel for
                    for (int nid = (1 << l) - 1; nid < (2 << (l + 1)) - 1; nid++) {
                        server.decrypt_gh(node_data[nid].sum_gh_pair);
                        node_data[nid].calc_weight(params.gbdt_param.lambda);
                    }
                    auto t2 = timer.now();
                    std::chrono::duration<float> t3 = t2 - t1;
                    decryption_time += t3.count();
                }

                #pragma omp parallel for
                for (int pid = 0; pid < parties.size(); pid++) {
                    for (int nid = (1 << l) - 1; nid < (1 << l) - 1 + n_nodes_in_level; nid++) {
                        server.send_node(nid, n_nodes_in_level, parties[pid]);
                    }
                }
                
                if (!split_further) {
                    // add Laplace noise to leaf node values
                    if (params.privacy_tech == "dp") {
                        for (int party_id = 0; party_id < parties.size(); party_id++) {
                            int tree_size = parties[party_id].booster.fbuilder->trees.nodes.size();
                            auto nodes = parties[party_id].booster.fbuilder->trees.nodes.host_data();
                            for (int node_id = 0; node_id < tree_size; node_id++) {
                                Tree::TreeNode node = nodes[node_id];
                                if (node.is_leaf) {
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
                // LOG(INFO)<<"tree nodes:"<<"\n"<<parties[pid].booster.fbuilder->trees.nodes;
                parties[pid].booster.fbuilder->predict_in_training(t);
            }
            server.booster.fbuilder->trees.prune_self(model_param.gamma);
            server.booster.fbuilder->predict_in_training(t);
            tree = parties[0].booster.fbuilder->trees;
//            tree.nodes.resize(parties[0].booster.fbuilder->trees.nodes.size());
//            tree.nodes.copy_from(parties[0].booster.fbuilder->trees.nodes);
        }

//        parties[0].gbdt.trees.push_back(trees);
        #pragma omp parallel for
        for (int p = 0; p < n_parties; p++) {
//            parties[p].gbdt.trees.push_back(parties_trees[p]);
            parties[p].gbdt.trees.push_back(trees);
        }
        server.global_trees.trees.push_back(trees);

//        LOG(INFO) << parties[0].booster.metric->get_name() << " = "
//                  << parties[0].booster.metric->get_score(parties[0].booster.fbuilder->get_y_predict());

        LOG(INFO) << server.booster.metric->get_name() << " = "
                  << server.booster.metric->get_score(server.booster.fbuilder->get_y_predict());

        LOG(INFO) << "Training round " << round << " end";
    }

    auto stop = timer.now();
    std::chrono::duration<float> training_time = stop - start;
    LOG(INFO) << "training time = " << training_time.count() << "s";
    if (params.privacy_tech == "he"){
        LOG(INFO) << "HE time (encryption and decryption) " << encryption_time + decryption_time << "("
                  << encryption_time << "/" << decryption_time << ")";
    }

    LOG(INFO) << "end of training";
}

// in this case, server is also a party which holds both the features and the label
void FLtrainer::vertical_fl_trainer_label_at_party(vector<Party> &parties, Server &server, FLParam &params) {
    std::chrono::high_resolution_clock timer;
    auto start = timer.now();
    float encryption_time = 0.0f;
    float decryption_time = 0.0f;
    // load dataset
    GBDTParam &model_param = params.gbdt_param;
    Comm comm_helper;
    DifferentialPrivacy dp_manager;
    int n_parties = parties.size();
    // initializing differential privacy
    if (params.privacy_tech == "dp") {
        dp_manager = DifferentialPrivacy();
        dp_manager.init(params);
    }
//    if(params.ins_bagging_fraction < 1.0){
//        LOG(INFO)<<"start bagging init";
//        for(int i = 0; i < n_parties; i++){
//            parties[i].bagging_init(36);
//        }
//        server.bagging_init(36);
//    }
    // start training
    // for each boosting round

    for (int round = 0; round < params.gbdt_param.n_trees; round++) {
//        LOG(INFO) << "Training round " << round << " start";

        vector<Tree> trees(params.gbdt_param.tree_per_rounds);

        if(params.ins_bagging_fraction < 1.0){
            if(round % (int(1/params.ins_bagging_fraction)) == 0) {
                for(int i = 0; i < n_parties; i++){
                    parties[i].bagging_init(36);
                }
                server.bagging_init(36);
            }
            server.sample_data();
            server.booster.init(server.dataset, params.gbdt_param);
            if(round!=0){
                server.predict_raw_vertical_jointly_in_training(params.gbdt_param, parties,
                                                                server.booster.fbuilder->get_y_predict());
            }
            for(int pid = 0; pid < n_parties; pid++) {
                parties[pid].sample_data();
                parties[pid].booster.init(parties[pid].dataset, params.gbdt_param);
            }
        }
        // Server update, encrypt and send gradients
        server.booster.update_gradients();
        // for the parties that have labels, they can compute histograms on unencrypted gradients
        for(int pid = 0; pid < n_parties; pid++) {
            if(parties[pid].has_label) {
                parties[pid].booster.set_gradients(server.booster.gradients);
            }
        }

        if (params.privacy_tech == "dp") {
            // option 1: direct add noise to gradients
//            server.booster.add_noise_to_gradients(params.variance);

            // option 2: clip gradients to (-1, 1)
            auto gradient_data = server.booster.gradients.host_data();
            for (int i = 0; i < server.booster.gradients.size(); i++) {
                dp_manager.clip_gradient_value(gradient_data[i].g);
            }
        }
        // temp_gradients store the raw gradients
        SyncArray<GHPair> temp_gradients;
        if (params.privacy_tech == "he") {
            auto t1 = timer.now();
            temp_gradients.resize(server.booster.gradients.size());
            temp_gradients.copy_from(server.booster.gradients);
            server.homo_init();
            server.encrypt_gh_pairs(server.booster.gradients);
            auto t2 = timer.now();
            std::chrono::duration<float> t3 = t2 - t1;
            encryption_time += t3.count();
        }
#pragma omp parallel for
        for (int j = 0; j < parties.size(); j++) {
            if(!parties[j].has_label)
                server.send_booster_gradients(parties[j]);
        }
        if (params.privacy_tech == "he") {
            server.booster.gradients.copy_from(temp_gradients);
        }
        // for each tree in a round
        for (int t = 0; t < params.gbdt_param.tree_per_rounds; t++) {
            Tree &tree = trees[t];

            // server and each party initialize ins2node_id, gradients, etc.
            server.booster.fbuilder->build_init(server.booster.gradients, t);

#pragma omp parallel for
            for (int pid = 0; pid < parties.size(); pid++)
                parties[pid].booster.fbuilder->build_init(parties[pid].booster.gradients, t);


            // for each level
            for (int l = 0; l < params.gbdt_param.depth; l++) {

                // initialize level parameters
                bool split_further=false;
                int n_nodes_in_level = 1 << l;
                int n_max_nodes = 2 << model_param.depth;
                vector<int> parties_n_bins(parties.size());
                vector<int> parties_n_columns(parties.size());
                MSyncArray<GHPair> parties_missing_gh(parties.size());
                MSyncArray<int> parties_hist_fid(parties.size());
                MSyncArray<int> parties_global_hist_fid(parties.size());
                MSyncArray<GHPair> parties_hist(parties.size());

                // each party computes hist, sends hist to server
                for (int pid = 0; pid < parties.size(); pid++)
                    parties_n_columns[pid] = parties[pid].dataset.n_features();

#pragma omp parallel for
                for (int pid = 0; pid < parties.size(); pid++) {
                    int n_bins = parties[pid].booster.fbuilder->cut.cut_points_val.size();
//                    std::cout<<"party "<<pid<<" n_bins:"<<n_bins<<std::endl;
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
                    SyncArray<GHPair> hist(n_nodes_in_level * n_bins);
                    parties[pid].booster.fbuilder->compute_histogram_in_a_level(l, n_max_splits, n_bins,
                                                                                n_nodes_in_level,
                                                                                hist_fid_data, missing_gh, hist);

                    parties_missing_gh[pid].resize(n_partition);
                    parties_missing_gh[pid].copy_from(missing_gh);
                    parties_hist[pid].resize(n_bins * n_nodes_in_level);
                    parties_hist[pid].copy_from(hist);

                    parties[pid].booster.fbuilder->sp.resize(n_nodes_in_level);
                }

                server.booster.fbuilder->sp.resize(n_nodes_in_level);
                // server concat hist_fid_data, missing_gh & histograms
                int n_bins_new = accumulate(parties_n_bins.begin(), parties_n_bins.end(), 0);
                int n_max_splits_new = n_max_nodes * n_bins_new;
                int n_column_new = accumulate(parties_n_columns.begin(), parties_n_columns.end(), 0);
                SyncArray<int> global_hist_fid(n_bins_new * n_nodes_in_level);
                SyncArray<GHPair> missing_gh(n_column_new * n_nodes_in_level);
                SyncArray<GHPair> hist(n_bins_new * n_nodes_in_level);
                global_hist_fid.copy_from(
                        comm_helper.concat_msyncarray(parties_global_hist_fid, n_nodes_in_level));

                // server compute gain
                SyncArray<float_type> gain(n_max_splits_new);
                if (params.privacy_tech == "he") {
                    auto t1 = timer.now();
                    for(int i = 0; i < n_parties; i++){
                        if(!parties[i].has_label){
                            server.decrypt_gh_pairs(parties_hist[i]);
                            server.decrypt_gh_pairs(parties_missing_gh[i]);
                        }
                    }
//                    server.decrypt_gh_pairs(hist);
//                    server.decrypt_gh_pairs(missing_gh);
                    auto t2 = timer.now();
                    std::chrono::duration<float> t3 = t2 - t1;
                    decryption_time += t3.count();
                }

                missing_gh.copy_from(
                        comm_helper.concat_msyncarray(parties_missing_gh, n_nodes_in_level));
                hist.copy_from(comm_helper.concat_msyncarray(parties_hist, n_nodes_in_level));

                // LOG(INFO) << "hist:"<<"\n"<<hist;
                server.booster.fbuilder->compute_gain_in_a_level(gain, n_nodes_in_level, n_bins_new,
                                                                 global_hist_fid.host_data(),
                                                                 missing_gh, hist);
                // server find the best gain and its index
                SyncArray<int_float> best_idx_gain(n_nodes_in_level);

                // with Exponential Mechanism: select with split probability
                if (params.privacy_tech == "dp") {
                    SyncArray<float_type> prob_exponent(
                            n_max_splits_new);    //the exponent of probability mass for each split point
                    dp_manager.compute_split_point_probability(gain, prob_exponent);
                    auto prob_exponent_data = prob_exponent.host_data();
                    dp_manager.exponential_select_split_point(prob_exponent, gain, best_idx_gain, n_nodes_in_level,
                                                              n_bins_new);
                }
                    // without Exponential Mechanism: select the split with max gain
                else {
                    server.booster.fbuilder->get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level,
                                                                      n_bins_new);
                }
                LOG(DEBUG) << "best_idx_gain:" << best_idx_gain;
                auto best_idx_data = best_idx_gain.host_data();

                // parties who propose the best candidate update their trees accordingly
                vector<vector<int>> party_node_map(parties.size());

                for (int node = 0; node < n_nodes_in_level; node++) {
                    auto server_nodes_data = server.booster.fbuilder->trees.nodes.host_data();
                    if (!server_nodes_data[node + n_nodes_in_level - 1].is_valid) {
                        continue;
                    }
                    // convert the global best index to party id & its local index
                    int best_idx = get<0>(best_idx_data[node]);
                    best_idx -= node * n_bins_new;
                    float best_gain = get<1>(best_idx_data[node]);
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

                    // LOG(INFO) << "sp:"<<"\n"<<parties[party_id].booster.fbuilder->sp;
                    // party update itself
                    parties[party_id].booster.fbuilder->update_tree_in_a_node(node);
                    parties[party_id].booster.fbuilder->update_ins2node_id_in_a_node(node_shifted);

                    // LOG(INFO)<<"level "<<l<<":tree nodes"<<"\n"<<parties[party_id].booster.fbuilder->trees.nodes;

                    if (!split_further) {
                        if (parties[party_id].booster.fbuilder->has_split) {
                            split_further = true;
                        }
                    }
                    // update local split_feature_id to global
                    auto party_global_hist_fid_data = parties_global_hist_fid[party_id].host_data();
                    int global_fid = party_global_hist_fid_data[local_idx];
                    auto nodes_data = parties[party_id].booster.fbuilder->trees.nodes.host_data();
                    auto sp_data = parties[party_id].booster.fbuilder->sp.host_data();
                    // LOG(INFO)<<"local split fea id:"<<sp_data[node].split_fea_id;
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
                    auto t1 = timer.now();
                    auto node_data = server.booster.fbuilder->trees.nodes.host_data();
#pragma omp parallel for
                    for (int nid = (1 << l) - 1; nid < (2 << (l + 1)) - 1; nid++) {
                        if(node_data[nid].sum_gh_pair.encrypted) {
                            server.decrypt_gh(node_data[nid].sum_gh_pair);
                            node_data[nid].calc_weight(params.gbdt_param.lambda);
                        }
                    }
                    auto t2 = timer.now();
                    std::chrono::duration<float> t3 = t2 - t1;
                    decryption_time += t3.count();
                }

#pragma omp parallel for
                for (int pid = 0; pid < parties.size(); pid++) {
                    for (int nid = (1 << l) - 1; nid < (1 << l) - 1 + n_nodes_in_level; nid++) {
                        server.send_node(nid, n_nodes_in_level, parties[pid]);
                    }
                }

                if (!split_further) {
                    // add Laplace noise to leaf node values
                    if (params.privacy_tech == "dp") {
                        for (int party_id = 0; party_id < parties.size(); party_id++) {
                            int tree_size = parties[party_id].booster.fbuilder->trees.nodes.size();
                            auto nodes = parties[party_id].booster.fbuilder->trees.nodes.host_data();
                            for (int node_id = 0; node_id < tree_size; node_id++) {
                                Tree::TreeNode node = nodes[node_id];
                                if (node.is_leaf) {
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
                // LOG(INFO)<<"tree nodes:"<<"\n"<<parties[pid].booster.fbuilder->trees.nodes;
                parties[pid].booster.fbuilder->predict_in_training(t);
            }
            server.booster.fbuilder->trees.prune_self(model_param.gamma);
            server.booster.fbuilder->predict_in_training(t);
            tree = parties[0].booster.fbuilder->trees;
//            tree.nodes.resize(parties[0].booster.fbuilder->trees.nodes.size());
//            tree.nodes.copy_from(parties[0].booster.fbuilder->trees.nodes);
        }

//        parties[0].gbdt.trees.push_back(trees);
#pragma omp parallel for
        for (int p = 0; p < n_parties; p++) {
//            parties[p].gbdt.trees.push_back(parties_trees[p]);
            parties[p].gbdt.trees.push_back(trees);
        }
        server.global_trees.trees.push_back(trees);

//        LOG(INFO) << parties[0].booster.metric->get_name() << " = "
//                  << parties[0].booster.metric->get_score(parties[0].booster.fbuilder->get_y_predict());

        LOG(INFO) << server.booster.metric->get_name() << " = "
                  << server.booster.metric->get_score(server.booster.fbuilder->get_y_predict());

//        LOG(INFO) << "Training round " << round << " end";
    }

    auto stop = timer.now();
    std::chrono::duration<float> training_time = stop - start;
    LOG(INFO) << "training time = " << training_time.count() << "s";
    if (params.privacy_tech == "he"){
        LOG(INFO) << "HE time (encryption and decryption) " << encryption_time + decryption_time << "("
                  << encryption_time << "/" << decryption_time << ")";
    }

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
    #pragma omp parallel for
    for (int i = 0; i < n_party; i++) {
        LOG(INFO)<<"In Party "<<i;
        for (int j = 0; j < params.gbdt_param.n_trees; j++)
            parties[i].booster.boost(parties[i].gbdt.trees);
    }
}
