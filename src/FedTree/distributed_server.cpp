//
// Created by 韩雨萱 on 11/4/21.
//

#include "FedTree/FL/distributed_server.h"
#include "FedTree/FL/partition.h"
#include "FedTree/parser.h"

grpc::Status DistributedServer::TriggerUpdateGradients(::grpc::ServerContext *context, const ::fedtree::PID *request,
                                                       ::fedtree::Ready *response) {
    booster.update_gradients();
    LOG(DEBUG) << "gradients updated";
    update_gradients_success = true;
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
    std::uniform_int_distribution<int> delay_distribution(10, 20);

    while (!update_gradients_success) {
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }

    int pid = id->id();  // useless in vertical training
    auto gh_data = booster.gradients.host_data();
    for (int i = 0; i < booster.gradients.size(); i++) {
        fedtree::GHPair gh;
        gh.set_g(gh_data[i].g);
        gh.set_h(gh_data[i].h);
        writer->Write(gh);
    }
    LOG(DEBUG) << "Send gradients to " << pid;
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
    std::uniform_int_distribution<int> delay_distribution(10, 20);

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

//    cont_votes.clear();
//    best_infos.clear();
    cur_round += 1;
    update_gradients_success = false;
    cont_votes.clear();
//    n_nodes_received = 0;

    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto itr = metadata.find("n_nodes_in_level");
    int n_nodes_in_level = std::stoi(itr->second.data());

    booster.fbuilder->sp.resize(n_nodes_in_level);
    int n_bins_new = accumulate(n_bins_per_party.begin(), n_bins_per_party.end(), 0);
    int n_column_new = accumulate(n_columns_per_party.begin(), n_columns_per_party.end(), 0);
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

    Comm comm_helper;
    hist_fid.copy_from(comm_helper.concat_msyncarray(booster.fbuilder->parties_hist_fid,
                                                     n_bins_per_party, n_nodes_in_level));

    missing_gh.copy_from(comm_helper.concat_msyncarray(booster.fbuilder->parties_missing_gh,
                                                       n_columns_per_party, n_nodes_in_level));

    hist.copy_from(comm_helper.concat_msyncarray(booster.fbuilder->parties_hist,
                                                 n_bins_per_party, n_nodes_in_level));

    SyncArray<float_type> gain(n_max_splits_new);

    booster.fbuilder->compute_gain_in_a_level(gain, n_nodes_in_level, n_bins_new, hist_fid.host_data(), missing_gh,
                                              hist, n_column_new);
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
        float best_gain = thrust::get<1>(best_idx_data[node]);
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
    std::uniform_int_distribution<int> delay_distribution(10, 20);
    while (!aggregate_success) {
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }
    int pid = id->id();
    for (const BestInfo &best_info : best_infos) {
        fedtree::BestInfo best;
        best.set_nid(best_info.nid);
        best.set_pid(best_info.pid);
        best.set_idx(best_info.idx);
        best.set_global_fid(best_info.global_fid);
        best.set_gain(best_info.gain);
        writer->Write(best);
    }
    std::cout << "Send best info to " << pid << std::endl;
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
    n_nodes_received += 1;
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
    std::uniform_int_distribution<int> delay_distribution(10, 20);

    while (n_nodes_received < n_nodes_in_level) {
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
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
    std::cout << "Send nodes to " << id->id() << std::endl;
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
    std::cout << "Send ins2node_id to " << id->id() << std::endl;
    return grpc::Status::OK;
}

grpc::Status DistributedServer::CheckIfContinue(grpc::ServerContext *context, const fedtree::PID *pid,
                                                fedtree::Ready *ready) {
    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();
    auto itr = metadata.find("cont");
    int cont = std::stoi(itr->second.data());
    cont_votes.push_back(cont);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> delay_distribution(10, 20);

    while (cont_votes.size() < param.n_parties) {
        std::this_thread::sleep_for(
                std::chrono::milliseconds(delay_distribution(generator)));
    }

    aggregate_success = false;
    best_infos.clear();
    n_nodes_received = 0;

    ready->set_ready(false);

    LOG(DEBUG) << "cont votes: " << cont_votes;

    for (auto vote:cont_votes) {
        if (vote) {
            ready->set_ready(true);
            LOG(DEBUG) << "agreed to continue";
            break;
        }
    }

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

void DistributedServer::InitVectors(int n_parties) {
    hists_received.resize(n_parties, 0);
    missing_gh_received.resize(n_parties, 0);
    hist_fid_received.resize(n_parties, 0);
}

void RunServer(DistributedServer &service) {
    std::string server_address("0.0.0.0:50051");

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    LOG(DEBUG) << "Server listening on " << server_address;
    server->Wait();
}

int main(int argc, char **argv) {
    int pid;
    FLParam fl_param;
    Parser parser;
    if (argc > 1) {
        parser.parse_param(fl_param, argc, argv);
    } else {
        printf("Usage: <config file path> \n");
        exit(0);
    }
    GBDTParam &model_param = fl_param.gbdt_param;
    DataSet dataset;
    dataset.load_from_file(model_param.path, fl_param);

    GBDTParam &param = fl_param.gbdt_param;
    if (param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos) {
        dataset.group_label();
        int num_class = dataset.label.size();
        if (param.num_class != num_class) {
            LOG(DEBUG) << "updating number of classes from " << param.num_class << " to " << num_class;
            param.num_class = num_class;
        }
        if (param.num_class > 2)
            param.tree_per_rounds = param.num_class;
    } else if (param.objective.find("reg:") != std::string::npos) {
        param.num_class = 1;
    }

    DistributedServer server;
    int n_parties = fl_param.n_parties;
    server.InitVectors(n_parties);
    vector<int> n_instances_per_party(n_parties);
    server.distributed_vertical_init(fl_param, dataset.n_instances(), n_instances_per_party, dataset.y);
    RunServer(server);
    return 0;
}