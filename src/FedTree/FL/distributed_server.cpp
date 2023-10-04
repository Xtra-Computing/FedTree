//
// Created by liqinbin on 8/22/22.
//

#include "FedTree/FL/distributed_server.h"
#include "FedTree/DP/differential_privacy.h"
#include "FedTree/FL/partition.h"
#include "FedTree/parser.h"
#include <cmath>
#include <mutex>
#include <sstream>

grpc::Status DistributedServer::TriggerUpdateGradients(::grpc::ServerContext *context, const ::fedtree::PID *request,
                                                       ::fedtree::Ready *response) {
//    LOG(INFO)<<"computation UpdateGradients start";
    booster.update_gradients();
    LOG(DEBUG) << "gradients updated";
    if (param.privacy_tech == "he") {
        // TIPS: store string

        SyncArray<GHPair> tmp;
        tmp.resize(booster.gradients.size());
        tmp.copy_from(booster.gradients);

        std::chrono::high_resolution_clock timer;
        auto t_start = timer.now();
        encrypt_gh_pairs(booster.gradients);
        auto t_end = timer.now();
        std::chrono::duration<double> used_time = t_end-t_start;
        enc_time += used_time.count();
        auto gh_data = booster.gradients.host_data();

        int len = booster.gradients.size();
        const int BATCH_SIZE = 5000;
        tmp_gradients.resize((len+BATCH_SIZE-1)/BATCH_SIZE);
        // #pragma omp parallel for
        for (int beg = 0; beg < len; beg+=BATCH_SIZE) {
            stringstream stream;
            int end = min(len, beg+BATCH_SIZE);
            fedtree::GHEncBatch ghb;
            tmp_gradients[beg/BATCH_SIZE].clear_g_enc();
            tmp_gradients[beg/BATCH_SIZE].clear_h_enc();

            for (int i = beg; i < end; i++) {
                stream << gh_data[i].g_enc;
                tmp_gradients[beg/BATCH_SIZE].add_g_enc(stream.str());
                stream.clear();
                stream.str("");
                stream << gh_data[i].h_enc;
                tmp_gradients[beg/BATCH_SIZE].add_h_enc(stream.str());
                stream.clear();
                stream.str("");
            }
        }

        booster.gradients.copy_from(tmp);
    }
    update_gradients_success = true;
//    LOG(INFO)<<"computation UpdateGradients end";
    return grpc::Status::OK;
}

grpc::Status DistributedServer::TriggerBuildInit(::grpc::ServerContext *context, const ::fedtree::PID *request,
                                                 ::fedtree::Ready *response) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto itr = metadata.find("t");
    int t = std::stoi(itr->second.data());
    booster.fbuilder->build_init(booster.gradients, t);
    LOG(DEBUG) << "build init";
    return grpc::Status::OK;
}

grpc::Status DistributedServer::GetGradients(grpc::ServerContext *context, const fedtree::PID *id,
                                             grpc::ServerWriter<fedtree::GHPair> *writer) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);
    int pid = id->id();  // useless in vertical training
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();
    while (!update_gradients_success) {
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    party_wait_times[pid] += used_time.count();

    auto gh_data = booster.gradients.host_data();
    for (int i = 0; i < booster.gradients.size(); i++) {
        fedtree::GHPair gh;
        gh.set_g(gh_data[i].g);
        gh.set_h(gh_data[i].h);
        writer->Write(gh);
    }
    LOG(DEBUG) << "Send " <<booster.gradients.size() <<" gradients to " << pid;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendDatasetInfo(grpc::ServerContext *context, const fedtree::DatasetInfo *datasetInfo,
                                                fedtree::PID *id) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto itr = metadata.find("pid");
    int pid = std::stoi(itr->second.data());
    n_bins_per_party[pid] = datasetInfo->n_bins();
    n_columns_per_party[pid] = datasetInfo->n_columns();
    LOG(DEBUG) << "receive dataset info from " << pid;
    return grpc::Status::OK;
}

grpc::Status
DistributedServer::SendHistograms(grpc::ServerContext *context, grpc::ServerReader<fedtree::GHPair> *reader,
                                  fedtree::PID *id) {
    fedtree::GHPair hist;
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());
    auto type_itr = metadata.find("type");
    int type = std::stoi(type_itr->second.data());

    vector<fedtree::GHPair> hists_vector;
    while (reader->Read(&hist)) {
        hists_vector.push_back(hist);
    }

    if (type == 0) {
        SyncArray<GHPair> &hists = booster.fbuilder->parties_hist[pid];
        hists.resize(hists_vector.size());
        auto hist_data = hists.host_data();
        for (int i = 0; i < hists_vector.size(); i++) {
            hist_data[i] = {static_cast<float_type>(hists_vector[i].g()), static_cast<float_type>(hists_vector[i].h())};
        }
    } else {
        SyncArray<GHPair> &hists = booster.fbuilder->parties_missing_gh[pid];
        hists.resize(hists_vector.size());
        auto hist_data = hists.host_data();
        for (int i = 0; i < hists_vector.size(); i++) {
            hist_data[i] = {static_cast<float_type>(hists_vector[i].g()), static_cast<float_type>(hists_vector[i].h())};
        }
    }

    LOG(DEBUG) << "Receive hist from " << pid;
    if (type == 0)
        hists_received[pid] += 1;
    else
        missing_gh_received[pid] += 1;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendHistFid(grpc::ServerContext *context, grpc::ServerReader<fedtree::FID> *reader,
                                            fedtree::PID *id) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());

    fedtree::FID fid;
    vector<int> hist_fid;
    while (reader->Read(&fid)) {
        hist_fid.push_back(fid.id());
    }

    SyncArray<int> &hists = booster.fbuilder->parties_hist_fid[pid];
    hists.resize(hist_fid.size());
    hists.copy_from(&hist_fid[0], hist_fid.size());

    LOG(DEBUG) << "Receive hist_fid from " << pid;
    hist_fid_received[pid] += 1;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::TriggerAggregate(grpc::ServerContext *context, const fedtree::PID *pid,
                                                 fedtree::Ready *ready) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);

    LOG(DEBUG) << hists_received << "/" << missing_gh_received << "/" << hist_fid_received;

    bool cont = true;
    while (cont) {
        cont = false;
        for (int i = 0; i < param.n_parties; ++i) {
            if (hists_received[i] < cur_round || missing_gh_received[i] < cur_round || hist_fid_received[i] < cur_round)
                cont = true;
        }
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }

//    best_infos.clear();
    cur_round += 1;
    update_gradients_success = false;
    n_nodes_received = 0;

    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto itr = metadata.find("n_nodes_in_level");
    int n_nodes_in_level = std::stoi(itr->second.data());

    booster.fbuilder->sp.resize(n_nodes_in_level);
    int n_total_bin = 0;
    vector<int> n_bins_per_party(param.n_parties);
    for(int i = 0; i < booster.fbuilder->parties_hist_fid.size(); i++){
        n_total_bin += booster.fbuilder->parties_hist_fid[i].size();
        n_bins_per_party[i] = booster.fbuilder->parties_hist_fid[i].size()/n_nodes_in_level;
    }
    int n_bins_new = n_total_bin / n_nodes_in_level;
    int n_total_column = 0;
    vector<int> n_columns_per_party(param.n_parties);
    for(int i = 0; i < booster.fbuilder->parties_missing_gh.size(); i++){
        n_total_column += booster.fbuilder->parties_missing_gh[i].size();
        n_columns_per_party[i] = booster.fbuilder->parties_missing_gh[i].size()/n_nodes_in_level;
    }
    int n_column_new = n_total_column / n_nodes_in_level;
//    int n_bins_new = accumulate(n_bins_per_party.begin(), n_bins_per_party.end(), 0);
//    int n_column_new = accumulate(n_columns_per_party.begin(), n_columns_per_party.end(), 0);
    int n_max_nodes = 2 << model_param.depth;
    int n_max_splits_new = n_max_nodes * n_bins_new;

    SyncArray<int> hist_fid(n_bins_new * n_nodes_in_level);
    SyncArray<GHPair> missing_gh(n_column_new * n_nodes_in_level);
    SyncArray<GHPair> hist(n_bins_new * n_nodes_in_level);

    for (int i = 1; i < param.n_parties; ++i) {
        int global_offset = accumulate(n_columns_per_party.begin(), n_columns_per_party.begin() + i, 0);
        auto hist_fid_data = booster.fbuilder->parties_hist_fid[i].host_data();
        for (int j = 0; j < booster.fbuilder->parties_hist_fid[i].size(); j++) {
            hist_fid_data[j] += global_offset;
        }
    }

    if (param.privacy_tech == "he") {
        std::chrono::high_resolution_clock timer;
        auto t1 = timer.now();
        for(int i = 0; i < param.n_parties; i++){
            if(!has_label[i]){
                decrypt_gh_pairs(booster.fbuilder->parties_hist[i]);
                decrypt_gh_pairs(booster.fbuilder->parties_missing_gh[i]);
            }
        }
        auto t2 = timer.now();
        std::chrono::duration<float> t3 = t2 - t1;
        dec_time += t3.count();
    }

    Comm comm_helper;
    hist_fid.copy_from(comm_helper.concat_msyncarray(booster.fbuilder->parties_hist_fid, n_nodes_in_level));

    missing_gh.copy_from(comm_helper.concat_msyncarray(booster.fbuilder->parties_missing_gh, n_nodes_in_level));

    hist.copy_from(comm_helper.concat_msyncarray(booster.fbuilder->parties_hist,n_nodes_in_level));

    SyncArray<float_type> gain(n_max_splits_new);


    booster.fbuilder->compute_gain_in_a_level(gain, n_nodes_in_level, n_bins_new, hist_fid.host_data(), missing_gh,
                                              hist);
    LOG(DEBUG) << "gain: " << gain;

    SyncArray<int_float> best_idx_gain(n_nodes_in_level);
    booster.fbuilder->get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins_new);
    auto best_idx_data = best_idx_gain.host_data();
    auto hist_fid_data = hist_fid.host_data();

    for (int node = 0; node < n_nodes_in_level; node++) {
        // convert the global best index to party id & its local index
        int best_idx = thrust::get<0>(best_idx_data[node]);
        int global_fid = hist_fid_data[best_idx];
        best_idx -= node * n_bins_new;
        float_type best_gain = thrust::get<1>(best_idx_data[node]);
        int party_id = 0;
        while (best_idx >= 0) {
            best_idx -= n_bins_per_party[party_id];
            party_id += 1;
        }
        party_id -= 1;
        int local_idx = best_idx + n_bins_per_party[party_id] * (node + 1);
        best_infos.push_back({party_id, node, local_idx, global_fid, best_gain});
    }

    LOG(DEBUG) << "best info: " << best_infos;

    aggregate_success = true;
    ready->set_ready(true);
    return grpc::Status::OK;
}

grpc::Status DistributedServer::GetBestInfo(grpc::ServerContext *context, const fedtree::PID *id,
                                            grpc::ServerWriter<fedtree::BestInfo> *writer) {

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);
    int pid = id->id();
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();
    while (!aggregate_success) {
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    party_wait_times[pid] += used_time.count();
    for (const BestInfo &best_info : best_infos) {
        fedtree::BestInfo best;
        best.set_nid(best_info.nid);
        best.set_pid(best_info.pid);
        best.set_idx(best_info.idx);
        best.set_global_fid(best_info.global_fid);
        best.set_gain(best_info.gain);
        writer->Write(best);
    }
    mutex.lock();
    info_cnt = (info_cnt + 1) % param.n_parties;
    if (info_cnt == 0) {
        aggregate_success = false;
        best_infos.clear();
    }
    mutex.unlock();
    LOG(DEBUG) << "Send best info to " << pid;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendNode(grpc::ServerContext *context, const fedtree::Node *node, fedtree::PID *id) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());

    int nid = node->final_id();
    auto nodes_data = booster.fbuilder->trees.nodes.host_data();
    nodes_data[nid].final_id = node->final_id();
    nodes_data[nid].lch_index = node->lch_index();
    nodes_data[nid].rch_index = node->rch_index();
    nodes_data[nid].parent_index = node->parent_index();
    nodes_data[nid].gain = node->gain();
    nodes_data[nid].base_weight = node->base_weight();
    nodes_data[nid].split_feature_id = node->split_feature_id();
    nodes_data[nid].pid = node->pid();
    nodes_data[nid].split_value = node->split_value();
    nodes_data[nid].split_bid = node->split_bid();
    nodes_data[nid].default_right = node->default_right();
    nodes_data[nid].is_leaf = node->is_leaf();
    nodes_data[nid].is_valid = node->is_valid();
    nodes_data[nid].is_pruned = node->is_pruned();
    nodes_data[nid].sum_gh_pair.g = node->sum_gh_pair_g();
    nodes_data[nid].sum_gh_pair.h = node->sum_gh_pair_h();
    nodes_data[nid].n_instances = node->n_instances();

    LOG(DEBUG) << "Receive node info from " << pid;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendNodes(grpc::ServerContext *context, const fedtree::NodeArray *node, fedtree::PID *id) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());
    int len = node->final_id_size();
    for (int i = 0; i < len; i++) {
        int nid = node->final_id(i);
        auto nodes_data = booster.fbuilder->trees.nodes.host_data();
        nodes_data[nid].final_id = node->final_id(i);
        nodes_data[nid].lch_index = node->lch_index(i);
        nodes_data[nid].rch_index = node->rch_index(i);
        nodes_data[nid].parent_index = node->parent_index(i);
        nodes_data[nid].gain = node->gain(i);
        nodes_data[nid].base_weight = node->base_weight(i);
        nodes_data[nid].split_feature_id = node->split_feature_id(i);
        nodes_data[nid].pid = node->pid(i);
        nodes_data[nid].split_value = node->split_value(i);
        nodes_data[nid].split_bid = node->split_bid(i);
        nodes_data[nid].default_right = node->default_right(i);
        nodes_data[nid].is_leaf = node->is_leaf(i);
        nodes_data[nid].is_valid = node->is_valid(i);
        nodes_data[nid].is_pruned = node->is_pruned(i);
        nodes_data[nid].sum_gh_pair.g = node->sum_gh_pair_g(i);
        nodes_data[nid].sum_gh_pair.h = node->sum_gh_pair_h(i);
        nodes_data[nid].n_instances = node->n_instances(i);
    }

    LOG(DEBUG) << "Receive node info from " << pid;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendNodesEnc(grpc::ServerContext *context, const fedtree::NodeEncArray *node, fedtree::PID *id) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());
    int len = node->final_id_size();
    for (int i = 0; i < len; i++) {
        int nid = node->final_id(i);
        auto nodes_data = booster.fbuilder->trees.nodes.host_data();
        nodes_data[nid].final_id = node->final_id(i);
        nodes_data[nid].lch_index = node->lch_index(i);
        nodes_data[nid].rch_index = node->rch_index(i);
        nodes_data[nid].parent_index = node->parent_index(i);
        nodes_data[nid].gain = node->gain(i);
        nodes_data[nid].base_weight = node->base_weight(i);
        nodes_data[nid].split_feature_id = node->split_feature_id(i);
        nodes_data[nid].pid = node->pid(i);
        nodes_data[nid].split_value = node->split_value(i);
        nodes_data[nid].split_bid = node->split_bid(i);
        nodes_data[nid].default_right = node->default_right(i);
        nodes_data[nid].is_leaf = node->is_leaf(i);
        nodes_data[nid].is_valid = node->is_valid(i);
        nodes_data[nid].is_pruned = node->is_pruned(i);
        nodes_data[nid].sum_gh_pair.encrypted = true;
        nodes_data[nid].sum_gh_pair.g_enc = NTL::to_ZZ(node->sum_gh_pair_g_enc(i).c_str());
        nodes_data[nid].sum_gh_pair.h_enc = NTL::to_ZZ(node->sum_gh_pair_h_enc(i).c_str());
        nodes_data[nid].sum_gh_pair.paillier = paillier;
        nodes_data[nid].n_instances = node->n_instances(i);
    }

    LOG(DEBUG) << "Receive node info from " << pid;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendNodeEnc(grpc::ServerContext *context, const fedtree::NodeEnc *node, fedtree::PID *id) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());

    int nid = node->final_id();
    auto nodes_data = booster.fbuilder->trees.nodes.host_data();
    nodes_data[nid].final_id = node->final_id();
    nodes_data[nid].lch_index = node->lch_index();
    nodes_data[nid].rch_index = node->rch_index();
    nodes_data[nid].parent_index = node->parent_index();
    nodes_data[nid].gain = node->gain();
    nodes_data[nid].base_weight = node->base_weight();
    nodes_data[nid].split_feature_id = node->split_feature_id();
    nodes_data[nid].pid = node->pid();
    nodes_data[nid].split_value = node->split_value();
    nodes_data[nid].split_bid = node->split_bid();
    nodes_data[nid].default_right = node->default_right();
    nodes_data[nid].is_leaf = node->is_leaf();
    nodes_data[nid].is_valid = node->is_valid();
    nodes_data[nid].is_pruned = node->is_pruned();
    nodes_data[nid].sum_gh_pair.encrypted = true;
    nodes_data[nid].sum_gh_pair.g_enc = NTL::to_ZZ(node->sum_gh_pair_g_enc().c_str());
    nodes_data[nid].sum_gh_pair.h_enc = NTL::to_ZZ(node->sum_gh_pair_h_enc().c_str());
    nodes_data[nid].sum_gh_pair.paillier = paillier;
    nodes_data[nid].n_instances = node->n_instances();

    LOG(DEBUG) << "Receive enc node info from " << pid;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendIns2NodeID(grpc::ServerContext *context,
                                               grpc::ServerReader<fedtree::Ins2NodeID> *reader, fedtree::PID *id) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());

    fedtree::Ins2NodeID i2n;
    auto ins2node_id_data = booster.fbuilder->ins2node_id.host_data();

    while (reader->Read(&i2n)) {
        int iid = i2n.iid();
        int nid = i2n.nid();
        ins2node_id_data[iid] = nid;
    }

    LOG(DEBUG) << "Receive ins2node_id from " << pid;
    mutex.lock();
    n_nodes_received += 1;
    mutex.unlock();
    return grpc::Status::OK;
}

grpc::Status DistributedServer::GetNodes(grpc::ServerContext *context, const fedtree::PID *id,
                                         grpc::ServerWriter<fedtree::Node> *writer) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto itr = metadata.find("l");
    int l = std::stoi(itr->second.data());
    int n_nodes_in_level = 1 << l;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();

    while (n_nodes_received < n_nodes_in_level) {
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    party_wait_times[id->id()] += used_time.count();

    if (param.privacy_tech == "he") {
        auto t1 = timer.now();
        auto node_data = booster.fbuilder->trees.nodes.host_data();
#pragma omp parallel for
        for (int nid = (1 << l) - 1; nid < (2 << (l + 1)) - 1; nid++) {
            if(node_data[nid].sum_gh_pair.encrypted) {
                decrypt_gh(node_data[nid].sum_gh_pair);
                node_data[nid].calc_weight(param.gbdt_param.lambda);
            }
        }
        auto t2 = timer.now();
        std::chrono::duration<float> t3 = t2 - t1;
        dec_time += t3.count();
    }
    auto nodes_data = booster.fbuilder->trees.nodes.host_data();
    for (int i = (1 << l) - 1; i < (4 << l) - 1; i++) {
        fedtree::Node node;
        node.set_final_id(nodes_data[i].final_id);
        node.set_lch_index(nodes_data[i].lch_index);
        node.set_rch_index(nodes_data[i].rch_index);
        node.set_parent_index(nodes_data[i].parent_index);
        node.set_gain(nodes_data[i].gain);
        node.set_base_weight(nodes_data[i].base_weight);
        node.set_split_feature_id(nodes_data[i].split_feature_id);
        node.set_pid(nodes_data[i].pid);
        node.set_split_value(nodes_data[i].split_value);
        node.set_split_bid(nodes_data[i].split_bid);
        node.set_default_right(nodes_data[i].default_right);
        node.set_is_leaf(nodes_data[i].is_leaf);
        node.set_is_valid(nodes_data[i].is_valid);
        node.set_is_pruned(nodes_data[i].is_pruned);
        node.set_sum_gh_pair_g(nodes_data[i].sum_gh_pair.g);
        node.set_sum_gh_pair_h(nodes_data[i].sum_gh_pair.h);
        node.set_n_instances(nodes_data[i].n_instances);
        writer->Write(node);
    }
    LOG(DEBUG) << "Send nodes to " << id->id();
    return grpc::Status::OK;
}

grpc::Status DistributedServer::GetIns2NodeID(grpc::ServerContext *context, const fedtree::PID *id,
                                              grpc::ServerWriter<fedtree::Ins2NodeID> *writer) {
    auto ins2node_id_data = booster.fbuilder->ins2node_id.host_data();
    for (int i = 0; i < booster.fbuilder->ins2node_id.size(); i++) {
        fedtree::Ins2NodeID i2n;
        i2n.set_iid(i);
        i2n.set_nid(ins2node_id_data[i]);
        writer->Write(i2n);
    }
    LOG(DEBUG) << "Send ins2node_id to " << id->id();
    return grpc::Status::OK;
}

grpc::Status DistributedServer::CheckIfContinue(grpc::ServerContext *context, const fedtree::PID *pid,
                                                fedtree::Ready *ready) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto itr = metadata.find("cont");
    int cont = std::stoi(itr->second.data());
    mutex.lock();
    cont_votes.push_back(cont);
    mutex.unlock();

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);

    while (cont_votes.size() < param.n_parties) {
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }

    // aggregate_success = false;
    // best_infos.clear();

    ready->set_ready(false);

    LOG(DEBUG) << "cont votes: " << cont_votes;

    for (auto vote:cont_votes) {
        if (vote) {
            ready->set_ready(true);
            LOG(DEBUG) << "agreed to continue";
            break;
        }
    }
    mutex.lock();
    vote_cnt = (vote_cnt + 1) % param.n_parties;
    if (vote_cnt == 0) {
        cont_votes.clear();
    }
    mutex.unlock();
    return grpc::Status::OK;
}

grpc::Status DistributedServer::TriggerPrune(grpc::ServerContext *context, const fedtree::PID *pid,
                                             fedtree::Ready *ready) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto itr = metadata.find("t");
    int t = std::stoi(itr->second.data());
    booster.fbuilder->trees.prune_self(param.gbdt_param.gamma);
    booster.fbuilder->predict_in_training(t);
    return grpc::Status::OK;
}

grpc::Status DistributedServer::TriggerPrintScore(grpc::ServerContext *context, const fedtree::PID *pid, fedtree::Ready *ready) {
    LOG(INFO) << booster.metric->get_name() << " = "
              << booster.metric->get_score(booster.fbuilder->get_y_predict());
    return grpc::Status::OK;
}

void DistributedServer::VerticalInitVectors(int n_parties) {
    hists_received.resize(n_parties, 0);
    missing_gh_received.resize(n_parties, 0);
    hist_fid_received.resize(n_parties, 0);
    party_wait_times.resize(n_parties, 0);
    stoppable.resize(n_parties, true);
    party_tot_times.resize(n_parties, 0);
    party_comm_times.resize(n_parties, 0);
    party_enc_times.resize(n_parties, 0);
}

void DistributedServer::HorizontalInitVectors(int n_parties) {
    // TODO
    range_received.resize(n_parties, 0);
    party_feature_range.resize(n_parties);

    hist_fid_received.resize(n_parties, 0);
    hists_received.resize(n_parties, 0);
    missing_gh_received.resize(n_parties, 0);

    party_gh_received.resize(n_parties, 0);
    party_ghs.resize(n_parties);

    score_received.resize(n_parties, 0);
    party_scores.resize(n_parties, 0);

    party_wait_times.resize(n_parties, 0);
    stoppable.resize(n_parties, true);
    party_tot_times.resize(n_parties, 0);
    party_comm_times.resize(n_parties, 0);
    party_enc_times.resize(n_parties, 0);

    party_DHKey_received.resize(n_parties, 0);
    party_noises_received.resize(n_parties, 0);

    parties_cut_points_received.resize(n_parties, 0);
}

grpc::Status DistributedServer::SendRange(grpc::ServerContext* context, grpc::ServerReader<fedtree::GHPair>* reader,
                                          fedtree::PID* response) {
    fedtree::GHPair frange;
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());
    party_feature_range[pid].clear();
    while(reader->Read(&frange)) {
        party_feature_range[pid].emplace_back(frange.g(), frange.h());
    }
    LOG(DEBUG) << "Receive range from " << pid;
    range_received[pid] += 1;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::TriggerCut(grpc::ServerContext* context, const fedtree::PID* request,
                                           fedtree::Ready* response) {
    // cyz: request is n_bins
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);
    while (true) {
        bool cont = true;
        for (int i = 0; i < param.n_parties; i++) {
            if (range_received[i] < 1) {
                cont = false;
            }
        }
        if (cont) {
            break;
        }
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    // cyz: g: min, h: max, party0 stores final result
    for (int i = 1; i < param.n_parties; i++) {
        for (int j = 0; j < party_feature_range[0].size(); j++) {
            if (party_feature_range[i][j].g < party_feature_range[0][j].g) {
                party_feature_range[0][j].g = party_feature_range[i][j].g;
            }
            if (party_feature_range[i][j].h > party_feature_range[0][j].h) {
                party_feature_range[0][j].h = party_feature_range[i][j].h;
            }
        }
    }

    vector<vector<float_type>> temp;
    for (auto e: party_feature_range[0]) {
        temp.push_back({e.g, e.h});
    }
    int n_bins = request->id();
    booster.fbuilder->cut.get_cut_points_by_feature_range(temp, n_bins);
    range_success = true;

    return grpc::Status::OK;
}

grpc::Status DistributedServer::GetRange(grpc::ServerContext* context, const fedtree::PID* request,
                                         grpc::ServerWriter<fedtree::GHPair>* writer) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();

    while (!range_success) {
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    party_wait_times[request->id()] += used_time.count();
    for(auto e: party_feature_range[0]) {
        fedtree::GHPair frange;
        frange.set_g(e.g);
        frange.set_h(e.h);
        writer->Write(frange);
    }
    LOG(DEBUG) << "Send range to " << request->id();
    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendGH(grpc::ServerContext* context, const fedtree::GHPair* request,
                                       fedtree::PID* response) {

    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());
    party_gh_received[pid] += 1;
    party_ghs[pid] = {(float_type)request->g(), (float_type)request->h()};
    LOG(DEBUG) << "Receive gh from " << pid;

    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);
    while (true) {
        bool cont = true;
        for (int i = 0; i < param.n_parties; i++) {
            if (party_gh_received[i] < gh_rounds) {
                cont = false;
            }
        }
        if (cont) {
            break;
        }
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    party_wait_times[pid] += used_time.count();
    mutex.lock();
    gh_cnt = (gh_cnt + 1) % param.n_parties;
    if (gh_cnt == 0) {
        gh_rounds += 1;
    }
    mutex.unlock();
    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendDHPubKey(grpc::ServerContext* context, const fedtree::DHPublicKey* request,
                                             fedtree::PID* response) {

    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());
    party_DHKey_received[pid] += 1;
    LOG(DEBUG) << "Receive DHPubKey from " << pid;
    if(dh.other_public_keys.length() != param.n_parties)
        dh.other_public_keys.SetLength(param.n_parties);
    dh.other_public_keys[pid] = NTL::to_ZZ(request->pk().c_str());
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);
    while (true) {
        bool cont = true;
        for (int i = 0; i < param.n_parties; i++) {
            if (party_DHKey_received[i] < 1) {
                cont = false;
            }
        }
        if (cont) {
            break;
        }
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    party_wait_times[pid] += used_time.count();
    return grpc::Status::OK;
}

grpc::Status DistributedServer::GetDHPubKeys(grpc::ServerContext *context, const fedtree::PID *id,
                                             grpc::ServerWriter<fedtree::DHPublicKeys> *writer) {
    fedtree::DHPublicKeys pk;
    for (int i = 0; i < dh.other_public_keys.length(); i++){
        stringstream stream;
        stream<<dh.other_public_keys[i];
        pk.add_pk(stream.str());
        stream.clear();
        stream.str("");
    }
    writer->Write(pk);
    LOG(DEBUG) << "Send DHPubKeys to " << id->id();
    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendNoises(grpc::ServerContext* context, const fedtree::SANoises* request, fedtree::PID* response){
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());
    party_noises_received[pid] += 1;
    LOG(DEBUG) << "Receive Noises from " << pid;
    if(dh.received_encrypted_noises.length() != param.n_parties * param.n_parties)
        dh.received_encrypted_noises.SetLength(param.n_parties * param.n_parties);
    for(int i = 0; i < param.n_parties; i++){
        dh.received_encrypted_noises[pid * param.n_parties + i] = NTL::to_ZZ(request->noises(i).c_str());
    }
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);
    while (true) {
        bool cont = true;
        for (int i = 0; i < param.n_parties; i++) {
            if (party_noises_received[i] < noise_rounds) {
                cont = false;
            }
        }
        if (cont) {
            break;
        }
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    party_wait_times[pid] += used_time.count();
    mutex.lock();
    noise_cnt = (noise_cnt + 1) % param.n_parties;
    if (noise_cnt == 0) {
        noise_rounds += 1;
    }
    mutex.unlock();
    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendCutPoints(grpc::ServerContext* context, const fedtree::CutPoints* request, fedtree::PID* response){
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());
    parties_cut_points_received[pid] += 1;
    LOG(DEBUG) << "Receive CutPoints from " << pid;

    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();
    int cut_points_val_size = request->cut_points_val_size();
    int cut_col_ptr_size = request->cut_col_ptr_size();
    int cut_fid_size = request->cut_fid_size();
    booster.fbuilder->parties_cut[pid].cut_points_val.resize(cut_points_val_size);
    booster.fbuilder->parties_cut[pid].cut_col_ptr.resize(cut_col_ptr_size);
    booster.fbuilder->parties_cut[pid].cut_fid.resize(cut_points_val_size);
    auto cut_points_val_data = booster.fbuilder->parties_cut[pid].cut_points_val.host_data();
    auto cut_col_ptr_data = booster.fbuilder->parties_cut[pid].cut_col_ptr.host_data();
    auto cut_fid_data = booster.fbuilder->parties_cut[pid].cut_fid.host_data();
    for(int i = 0; i < cut_points_val_size; i++){
        cut_points_val_data[i] = request->cut_points_val(i);
        cut_fid_data[i] = request->cut_fid(i);
    }
    for(int i = 0; i < cut_col_ptr_size; i++){
        cut_col_ptr_data[i] = request->cut_col_ptr(i);
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);
    while (true) {
        bool cont = true;
        for (int i = 0; i < param.n_parties; i++) {
            if (parties_cut_points_received[i] < 1) {
                cont = false;
            }
        }
        if (cont) {
            break;
        }
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    party_wait_times[pid] += used_time.count();
    booster.fbuilder->cut.get_cut_points_by_parties_cut_sampling(booster.fbuilder->parties_cut, param.gbdt_param.max_num_bin);
    return grpc::Status::OK;
}

grpc::Status DistributedServer::GetCutPoints(grpc::ServerContext *context, const fedtree::PID *id,
                                             grpc::ServerWriter<fedtree::CutPoints> *writer) {
    int pid = id->id();
    fedtree::CutPoints cp;
    auto cut_val_data = booster.fbuilder->cut.cut_points_val.host_data();
    auto cut_col_ptr_data = booster.fbuilder->cut.cut_col_ptr.host_data();
    auto cut_fid_data = booster.fbuilder->cut.cut_fid.host_data();
    for(int i = 0; i < booster.fbuilder->cut.cut_points_val.size(); i++){
        cp.add_cut_points_val(cut_val_data[i]);
        cp.add_cut_fid(cut_fid_data[i]);
    }
    for(int i = 0; i < booster.fbuilder->cut.cut_col_ptr.size(); i++){
        cp.add_cut_col_ptr(cut_col_ptr_data[i]);
    }
    writer->Write(cp);
    LOG(DEBUG) << "Send CutPoints to " << pid;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::GetNoises(grpc::ServerContext *context, const fedtree::PID *id,
                                          grpc::ServerWriter<fedtree::SANoises> *writer) {
    int pid = id->id();
    fedtree::SANoises pk;
    for (int i = 0; i < param.n_parties; i++){
        stringstream stream;
        stream<<dh.received_encrypted_noises[i * param.n_parties + pid];
        pk.add_noises(stream.str());
        stream.clear();
        stream.str("");
    }
    writer->Write(pk);
    LOG(DEBUG) << "Send Noises to " << pid;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::TriggerBuildUsingGH(grpc::ServerContext* context, const fedtree::PID* request,
                                                    fedtree::Ready* response) {
    // TODO
    GHPair sum_gh;
    for (auto e: party_ghs) {
        sum_gh = sum_gh + e;
    }
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto k_itr = metadata.find("k");
    int k = std::stoi(k_itr->second.data());
    booster.fbuilder->build_init(sum_gh, k);
    build_gh_success = true;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::TriggerCalcTree(grpc::ServerContext* context, const fedtree::PID* request, fedtree::Ready* response) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);

    LOG(DEBUG) << hists_received << "/" << missing_gh_received << "/" << hist_fid_received;
    {
        TIMED_SCOPE(timerObj, "calc_wait");
        bool cont = true;
        while (cont) {
            cont = false;
            for (int i = 0; i < param.n_parties; ++i) {
                if (hists_received[i] < cur_round || missing_gh_received[i] < cur_round || hist_fid_received[i] < cur_round)
                    cont = true;
            }
            std::this_thread::sleep_for(
                    std::chrono::milliseconds(delay_distribution(generator)));
        }
    }
    cur_round += 1;
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto itr = metadata.find("l");
    int l = std::stoi(itr->second.data());
    int n_nodes_in_level = 1 << l;
    //
    SyncArray <GHPair> missing_gh;
    SyncArray <GHPair> hist;
    {
        TIMED_SCOPE(timerObj, "homo_add_time");
        booster.fbuilder->merge_histograms_server_propose(hist, missing_gh);
    }
    int n_bins = booster.fbuilder->cut.cut_points_val.size();
    int n_max_nodes = 2 << model_param.depth;
    int n_max_splits = n_max_nodes * n_bins;

    SyncArray <float_type> gain(n_max_splits);
    // if privacy tech == 'he', decrypt histogram
    if (param.privacy_tech == "he") {
        // TIMED_SCOPE(timerObj, "decrypting time");
        std::chrono::high_resolution_clock timer;
        auto t1 = timer.now();
        decrypt_gh_pairs(hist);
        decrypt_gh_pairs(missing_gh);
        auto t2 = timer.now();
        std::chrono::duration<double> t3 = t2-t1;
        dec_time += t3.count();
    }
    auto hist_fid_data = booster.fbuilder->parties_hist_fid[0].host_data();
    booster.fbuilder->compute_gain_in_a_level(gain, n_nodes_in_level, n_bins, hist_fid_data,
                                              missing_gh, hist);
    SyncArray <int_float> best_idx_gain(n_nodes_in_level);
    // if(params.privacy_tech == "dp"){
    //     SyncArray<float_type> prob_exponent(n_max_splits);    //the exponent of probability mass for each split point
    //     dp_manager.compute_split_point_probability(gain, prob_exponent);
    //     dp_manager.exponential_select_split_point(prob_exponent, gain, best_idx_gain, n_nodes_in_level, n_bins);
    // } else
    booster.fbuilder->get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins);
    booster.fbuilder->get_split_points(best_idx_gain, n_nodes_in_level, hist_fid_data, missing_gh, hist);
    // SP ready

    calc_success = true;
    booster.fbuilder->update_tree();
    return grpc::Status::OK;
}

grpc::Status DistributedServer::GetRootNode(grpc::ServerContext* context, const fedtree::PID *request, fedtree::Node* response) {
    auto & root_node = booster.fbuilder->trees.nodes.host_data()[0];
    response->set_final_id(root_node.final_id);
    response->set_lch_index(root_node.lch_index);
    response->set_rch_index(root_node.rch_index);
    response->set_parent_index(root_node.parent_index);
    response->set_gain(root_node.gain);
    response->set_base_weight(root_node.base_weight);
    response->set_split_feature_id(root_node.split_feature_id);
    response->set_pid(root_node.pid);
    response->set_split_value(root_node.split_value);
    response->set_split_bid(root_node.split_bid);
    response->set_default_right(root_node.default_right);
    response->set_is_leaf(root_node.is_leaf);
    response->set_is_valid(root_node.is_valid);
    response->set_is_pruned(root_node.is_pruned);
    response->set_sum_gh_pair_g(root_node.sum_gh_pair.g);
    response->set_sum_gh_pair_h(root_node.sum_gh_pair.h);
    response->set_n_instances(root_node.n_instances);
    return grpc::Status::OK;
}

grpc::Status DistributedServer::GetSplitPoints(grpc::ServerContext* context, const fedtree::PID* request,
                                               grpc::ServerWriter<fedtree::SplitPoint>* writer) {
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);
    while (!calc_success) {
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    party_wait_times[request->id()] += used_time.count();
    mutex.lock();

    sp_cnt = (sp_cnt + 1) % param.n_parties;
    if (sp_cnt == 0) {
        calc_success = false;
    }
    mutex.unlock();
    // SP
    auto &sp = booster.fbuilder->sp;
    auto sp_data = sp.host_data();
    for (int i = 0; i < sp.size(); i++) {
        fedtree::SplitPoint sp_point;
        sp_point.set_gain(sp_data[i].gain);
        sp_point.set_fea_missing_g(sp_data[i].fea_missing_gh.g);
        sp_point.set_fea_missing_h(sp_data[i].fea_missing_gh.h);
        sp_point.set_rch_sum_g(sp_data[i].rch_sum_gh.g);
        sp_point.set_rch_sum_h(sp_data[i].rch_sum_gh.h);
        sp_point.set_default_right(sp_data[i].default_right);
        sp_point.set_nid(sp_data[i].nid);
        sp_point.set_split_fea_id(sp_data[i].split_fea_id);
        sp_point.set_fval(sp_data[i].fval);
        sp_point.set_split_bid(sp_data[i].split_bid);
        sp_point.set_no_split_value_update(sp_data[i].no_split_value_update);
        writer->Write(sp_point);

    }
    LOG(DEBUG) << "Send split points to " << request->id();
    return grpc::Status::OK;
}

grpc::Status DistributedServer::HCheckIfContinue(grpc::ServerContext *context, const fedtree::PID *pid,
                                                 fedtree::Ready *ready) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto itr = metadata.find("cont");
    int cont = std::stoi(itr->second.data());
    mutex.lock();
    cont_votes.push_back(cont);
    mutex.unlock();
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);

    while (cont_votes.size() < param.n_parties) {
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    ready->set_ready(true);
    for (auto vote: cont_votes) {
        if (vote == 0) {
            ready->set_ready(false);
            break;
        }
    }
    mutex.lock();
    vote_cnt = (vote_cnt + 1) % param.n_parties;
    if (vote_cnt == 0) {
        cont_votes.clear();
    }
    mutex.unlock();
    return grpc::Status::OK;
}

grpc::Status DistributedServer::ScoreReduce(grpc::ServerContext* context, const fedtree::Score* request,
                                            fedtree::Score* response) {
    // emm
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());
    party_scores[pid] = request->content();
    score_received[pid] += 1;
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);
    while (true) {
        bool cont = true;
        for (int i = 0; i < param.n_parties; i++) {
            if (score_received[i] < score_rounds) {
                cont = false;
            }
        }
        if (cont) {
            break;
        }
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    party_wait_times[pid] += used_time.count();
    mutex.lock();
    cnt = (cnt + 1) % param.n_parties;
    if (cnt == 0) {
        score_rounds += 1;
    }
    mutex.unlock();
    float sum_score = 0;
    for (auto e: party_scores) {
        sum_score += e;
    }
    response->set_content(sum_score/param.n_parties);
    return grpc::Status::OK;
}

grpc::Status DistributedServer::TriggerHomoInit(grpc::ServerContext *context, const fedtree::PID *request,
                                                fedtree::Ready *response) {
//    LOG(INFO) << "computation HomoInit start";
    homo_init(param.key_length);
    homo_init_success = true;
//    LOG(INFO) << "computation HomoInit end";
    return grpc::Status::OK;
}

grpc::Status DistributedServer::TriggerSAInit(grpc::ServerContext *context, const fedtree::PID *request,
                                              fedtree::Ready *response) {
//    LOG(INFO) << "computation HomoInit start";
//    dh.primegen();
//    LOG(INFO) << "computation HomoInit end";
    return grpc::Status::OK;
}

grpc::Status DistributedServer::GetPaillier(grpc::ServerContext *context, const fedtree::PID *request,
                                            fedtree::Paillier * response) {
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);

    while (!homo_init_success) {
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    party_wait_times[request->id()] += used_time.count();

    stringstream stream;
    stream << paillier.modulus;
    response->set_modulus(stream.str());
    stream.clear();
    stream.str("");
    stream << paillier.generator;
    response->set_generator(stream.str());
    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendHistogramsEnc(grpc::ServerContext *context, grpc::ServerReader<fedtree::GHPairEnc> *reader,
                                                  fedtree::PID *id) {
    fedtree::GHPairEnc hist;
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());
    auto type_itr = metadata.find("type");
    int type = std::stoi(type_itr->second.data());

    vector<fedtree::GHPairEnc> hists_vector;
    while (reader->Read(&hist)) {
        hists_vector.push_back(hist);
    }
    if (type == 0) {
        SyncArray<GHPair> &hists = booster.fbuilder->parties_hist[pid];
        hists.resize(hists_vector.size());
        auto hist_data = hists.host_data();
        for (int i = 0; i < hists_vector.size(); i++) {
            hist_data[i].encrypted = true;
            hist_data[i].g_enc = NTL::to_ZZ(hists_vector[i].g_enc().c_str());
            hist_data[i].h_enc = NTL::to_ZZ(hists_vector[i].h_enc().c_str());
            hist_data[i].paillier = paillier;
        }
    } else {
        SyncArray<GHPair> &hists = booster.fbuilder->parties_missing_gh[pid];
        hists.resize(hists_vector.size());
        auto hist_data = hists.host_data();
        for (int i = 0; i < hists_vector.size(); i++) {
            hist_data[i].encrypted = true;
            hist_data[i].g_enc = NTL::to_ZZ(hists_vector[i].g_enc().c_str());
            hist_data[i].h_enc = NTL::to_ZZ(hists_vector[i].h_enc().c_str());
            hist_data[i].paillier = paillier;
        }
    }
    LOG(DEBUG) << "Receive encrypted hist from " << pid;
    if (type == 0)
        hists_received[pid] += 1;
    else
        missing_gh_received[pid] += 1;

    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendHistogramBatches(grpc::ServerContext *context, grpc::ServerReader<fedtree::GHBatch> *reader,
                                                     fedtree::PID *id) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());
    auto type_itr = metadata.find("type");
    int type = std::stoi(type_itr->second.data());
    fedtree::GHBatch tmp;
    fedtree::GHBatch all;
    while(reader->Read(&tmp)) {
        all.MergeFrom(tmp);
    }
    has_label[pid] = true;
    SyncArray<GHPair> &hists = (type == 0)? booster.fbuilder->parties_hist[pid]:booster.fbuilder->parties_missing_gh[pid];
    int len = all.g_size();
    hists.resize(len);
    auto hist_data = hists.host_data();
#pragma omp parallel for
    for (int i = 0; i < len; i++) {
        hist_data[i] = {static_cast<float_type>(all.g(i)), static_cast<float_type>(all.h(i))};
    }
    if (type == 0)
        hists_received[pid] += 1;
    else
        missing_gh_received[pid] += 1;

    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendHistFidBatches(grpc::ServerContext *context, grpc::ServerReader<fedtree::FIDBatch> *reader,
                                                   fedtree::PID *id) {

    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());

    fedtree::FIDBatch tmp;
    fedtree::FIDBatch all;
    while(reader->Read(&tmp)) {
        all.MergeFrom(tmp);
    }

    SyncArray<int> &hists = booster.fbuilder->parties_hist_fid[pid];
    int len = all.id_size();
    hists.resize(len);
    auto hist_data = hists.host_data();
    for (int i = 0; i < len; i++) {
        hist_data[i] = all.id(i);
    }

    LOG(DEBUG) << "Receive hist_fid from " << pid;
    hist_fid_received[pid] += 1;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::GetIns2NodeIDBatches(grpc::ServerContext *context, const fedtree::PID *id,
                                                     grpc::ServerWriter<fedtree::Ins2NodeIDBatch> *writer) {
    auto ins2node_id_data = booster.fbuilder->ins2node_id.host_data();
    const int BATCH_SIZE = 200000;
    int len = booster.fbuilder->ins2node_id.size();
    for (int beg = 0; beg < len; beg+=BATCH_SIZE) {
        int end = min(beg+BATCH_SIZE, len);
        fedtree::Ins2NodeIDBatch i2n;
        for (int i = beg; i < end; i++) {
            i2n.add_iid(i);
            i2n.add_nid(ins2node_id_data[i]);
        }
        writer->Write(i2n);
    }
    LOG(DEBUG) <<len<<" "<< "Send ins2node_id to " << id->id();

    return grpc::Status::OK;
}
grpc::Status DistributedServer::SendIns2NodeIDBatches(grpc::ServerContext *context, grpc::ServerReader<fedtree::Ins2NodeIDBatch> *reader,
                                                      fedtree::PID *id) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());

    fedtree::Ins2NodeIDBatch i2n;
    auto ins2node_id_data = booster.fbuilder->ins2node_id.host_data();

    while (reader->Read(&i2n)) {
        int len = i2n.iid_size();
        for (int i = 0; i < len; i++) {
            int iid = i2n.iid(i);
            int nid = i2n.nid(i);
            ins2node_id_data[iid] = nid;
        }
    }

    LOG(DEBUG) << "Receive ins2node_id batches from " << pid;
    mutex.lock();
    n_nodes_received += 1;
    mutex.unlock();
    return grpc::Status::OK;
}
grpc::Status DistributedServer::GetGradientBatches(grpc::ServerContext *context, const fedtree::PID *id,
                                                   grpc::ServerWriter<fedtree::GHBatch> *writer){
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);

    while (!update_gradients_success) {
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    int pid = id->id();  // useless in vertical training
    party_wait_times[pid] += used_time.count();

    auto gh_data = booster.gradients.host_data();
    int len = booster.gradients.size();
    const int BATCH_SIZE = 200000;
    for (int beg = 0; beg < len; beg+=BATCH_SIZE) {
        int end = min(len, beg+BATCH_SIZE);
        fedtree::GHBatch ghb;
        for (int i = beg; i < end; i++) {
            ghb.add_g(gh_data[i].g);
            ghb.add_h(gh_data[i].h);
        }
        writer->Write(ghb);
        // gh.set_g(gh_data[i].g);
        // gh.set_h(gh_data[i].h);
        // writer->Write(gh);
    }
    LOG(DEBUG) << "Send " <<booster.gradients.size() <<" gradients to " << pid << " by batching";
    return grpc::Status::OK;
}

grpc::Status DistributedServer::GetGradientBatchesEnc(grpc::ServerContext *context, const fedtree::PID *id,
                                                      grpc::ServerWriter<fedtree::GHEncBatch> *writer) {
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);
    while (!update_gradients_success) {
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    int pid = id->id();  // useless in vertical training
    party_wait_times[pid] += used_time.count();
    for (const auto &e: tmp_gradients) {
        if (!writer->Write(e)) {
            break;
        }
    }

    return grpc::Status::OK;
}

grpc::Status DistributedServer::SendHistogramBatchesEnc(grpc::ServerContext *context, grpc::ServerReader<fedtree::GHEncBatch> *reader,
                                                        fedtree::PID *id) {

    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto pid_itr = metadata.find("pid");
    int pid = std::stoi(pid_itr->second.data());
    auto type_itr = metadata.find("type");
    int type = std::stoi(type_itr->second.data());
    fedtree::GHEncBatch tmp;
    fedtree::GHEncBatch all;
    {
        TIMED_SCOPE(timerObj, "receive time");
        while(reader->Read(&tmp)) {
            all.MergeFrom(tmp);
        }
    }
    has_label[pid] = 0;
    SyncArray<GHPair> &hists = (type == 0)? booster.fbuilder->parties_hist[pid]:booster.fbuilder->parties_missing_gh[pid];
    int len = all.g_enc_size();
    hists.resize(len);
    auto hist_data = hists.host_data();
    {
        TIMED_SCOPE(timerObj, "decode time");
#pragma omp parallel for
        for (int i = 0; i < len; i++) {
            hist_data[i].encrypted = true;
            hist_data[i].g_enc = NTL::to_ZZ(all.g_enc(i).c_str());
            hist_data[i].h_enc = NTL::to_ZZ(all.h_enc(i).c_str());
            hist_data[i].paillier = paillier;
        }
    }
    if (type == 0)
        hists_received[pid] += 1;
    else
        missing_gh_received[pid] += 1;
    LOG(DEBUG) << "Receive encrypted hist batches from " << pid;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::StopServer(grpc::ServerContext *context, const fedtree::PID *request,
                                           fedtree::Score *ready) {
    stoppable[request->id()] = true;
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto tot_itr = metadata.find("tot");
    auto comm_itr = metadata.find("comm");
    auto enc_itr = metadata.find("enc");
    party_tot_times[request->id()] = std::stof(tot_itr->second.data());
    party_comm_times[request->id()] = std::stof(comm_itr->second.data());
    party_enc_times[request->id()] = std::stof(enc_itr->second.data());
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);
    while (true) {
        bool cont = true;
        for (int i = 0; i < param.n_parties; i++) {
            if (!stoppable[i]) {
                cont = false;
            }
        }
        if (cont) {
            break;
        }
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    ready->set_content(party_wait_times[request->id()]);

    if (request->id() == 0) {
        LOG(INFO) << "dec time: " << dec_time;
        LOG(INFO) << "wait times: " << party_wait_times;
        LOG(INFO) << "enc times: " << party_enc_times;
        float sum1 = 0, sum2 = 0;
        for (auto e: party_tot_times)
            sum1 += e;
        for (auto e: party_comm_times)
            sum2 += e;
        for (auto e: party_wait_times)
            sum2 -= e;
        vector<float> real_comm_times = party_comm_times;
        for (int i = 0; i < party_wait_times.size(); i++) {
            real_comm_times[i] -= party_wait_times[i];
        }
        LOG(INFO) << "train times: " << party_tot_times;
        LOG(INFO) << "real comm times: " << real_comm_times;

        float sum3 = 0;
        for (int i = 0; i < party_tot_times.size(); i++) {
            float tmp = party_tot_times[i] - sum1/param.n_parties;
            sum3 += tmp*tmp;
        }
        float sum4 = 0;
        for (int i = 0; i < real_comm_times.size(); i++) {
            float tmp = real_comm_times[i] - sum2/param.n_parties;
            sum4 += tmp*tmp;
        }
        float sum5 = 0;
        for (auto e: party_enc_times) {
            sum5 += e;
        }
        LOG(INFO) << "avg train time: " << sum1/param.n_parties << "s";
        LOG(INFO) << "train time stddev: " << sqrt(sum3/param.n_parties) << "s";
        LOG(INFO) << "avg real comm time: " << sum2/param.n_parties << "s";
        LOG(INFO) << "real comm time stddev: " << sqrt(sum4/param.n_parties) << "s";
        LOG(INFO) << "avg enc_dec time: " <<(sum5 / param.n_parties + dec_time + enc_time)<< "s";

        // Notify any thread waiting for the shutdown signal.
        std::unique_lock<std::mutex> guard(shutdown_lock);
        shutdown_ready = true;
        shutdown_cv.notify_all();
    }
    return grpc::Status::OK;
}

grpc::Status DistributedServer::BeginBarrier(grpc::ServerContext *context, const fedtree::PID *request,
                                             fedtree::Ready *ready) {
    stoppable[request->id()] = false;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(5, 10);
    while (true) {
        bool cont = true;
        for (int i = 0; i < param.n_parties; i++) {
            if (stoppable[i]) {
                cont = false;
            }
        }
        if (cont) {
            break;
        }
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    return grpc::Status::OK;
}

void DistributedServer::block_until_shutdown() {
    std::unique_lock<std::mutex> guard(this->shutdown_lock);
    this->shutdown_cv.wait(guard, [this] { return this->shutdown_ready; });
}
