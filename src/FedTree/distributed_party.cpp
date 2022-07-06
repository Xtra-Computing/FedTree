//
// Created by yuxuan on 11/4/21.
//

#include "FedTree/FL/distributed_party.h"
#include "FedTree/FL/partition.h"
#include "FedTree/parser.h"
#include "FedTree/DP/differential_privacy.h"

#include <sstream>
void DistributedParty::TriggerUpdateGradients() {
    fedtree::PID id;
    fedtree::Ready ready;
    grpc::ClientContext context;
    grpc::Status status = stub_->TriggerUpdateGradients(&context, id, &ready);
    if (status.ok()) {
        LOG(DEBUG) << "Triggered the server to update gradients.";
    } else {
        LOG(ERROR) << "TriggerUpdateGradients rpc failed.";
    }
}

void DistributedParty::TriggerBuildInit(int t) {
    fedtree::PID id;
    fedtree::Ready ready;
    grpc::ClientContext context;
    context.AddMetadata("t", std::to_string(t));
    grpc::Status status = stub_->TriggerBuildInit(&context, id, &ready);
    if (status.ok()) {
        LOG(DEBUG) << "Triggered the server to build init.";
    } else {
        LOG(ERROR) << "TriggerBuildInit rpc failed.";
    }
}

void DistributedParty::GetGradients() {
    fedtree::PID id;
    fedtree::GHPair gh;
    grpc::ClientContext context;
    auto t_start = timer.now();
    id.set_id(pid);
    LOG(DEBUG) << "Receiving gradients from the server.";
    auto booster_gradients_data = booster.gradients.host_data();
    int i = 0;
    std::unique_ptr<grpc::ClientReader<fedtree::GHPair> > reader(stub_->GetGradients(&context, id));

    while (reader->Read(&gh)) {
        booster_gradients_data[i] = {static_cast<float_type>(gh.g()), static_cast<float_type>(gh.h())};
        i++;
    }
    grpc::Status status = reader->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += i * 16 * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "All gradients received.";
    } else {
        LOG(ERROR) << "GetGradients rpc failed.";
    }
}

void DistributedParty::SendDatasetInfo(int n_bins, int n_columns) {
    fedtree::DatasetInfo datasetInfo;
    fedtree::PID id;
    grpc::ClientContext context;
    LOG(INFO)<<"communication MetaInfo start";
    auto t_start = timer.now();
    datasetInfo.set_n_bins(n_bins);
    datasetInfo.set_n_columns(n_columns);
    context.AddMetadata("pid", std::to_string(pid));

    grpc::Status status = stub_->SendDatasetInfo(&context, datasetInfo, &id);
    auto t_end = timer.now();
    LOG(INFO)<<"communication MetaInfo end";
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += datasetInfo.ByteSizeLong() * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "Dataset info sent.";
    } else {
        LOG(ERROR) << "SendDatasetInfo rpc failed.";
    }
}

void DistributedParty::SendHistograms(const SyncArray<GHPair> &hist, int type) {
    fedtree::PID id;
    grpc::ClientContext context;
    TIMED_SCOPE(timerObj, "SendHistogramsTime");
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    context.AddMetadata("type", std::to_string(type));

    std::unique_ptr<grpc::ClientWriter<fedtree::GHPair> > writer(
            stub_->SendHistograms(&context, &id));

    auto hist_data = hist.host_data();
    for (int i = 0; i < hist.size(); ++i) {
        fedtree::GHPair gh;
        gh.set_g(hist_data[i].g);
        gh.set_h(hist_data[i].h);
        if (!writer->Write(gh)) {
            break;
        }
    }
    writer->WritesDone();
    grpc::Status status = writer->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += hist.size() * 16 * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "All " << type << " sent.";
    } else {
        LOG(ERROR) << "SendHistograms rpc failed.";
    }
}

void DistributedParty::SendHistFid(const SyncArray<int> &hist_fid) {
    fedtree::PID id;
    grpc::ClientContext context;
    TIMED_SCOPE(timerObj, "SendHistFidTime");
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    std::unique_ptr<grpc::ClientWriter<fedtree::FID> > writer(
            stub_->SendHistFid(&context, &id));

    auto hist_fid_data = hist_fid.host_data();
    for (int i = 0; i < hist_fid.size(); ++i) {
        fedtree::FID fid;
        fid.set_id(hist_fid_data[i]);
        if (!writer->Write(fid)) {
            break;
        }
    }
    writer->WritesDone();
    grpc::Status status = writer->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += hist_fid.size() * 4 * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "All hist_fid sent.";
    } else {
        LOG(ERROR) << "SendHistograms rpc failed.";
    }
}

bool DistributedParty::TriggerAggregate(int n_nodes_in_level) {
    fedtree::PID id;
    fedtree::Ready ready;
    id.set_id(pid);
    grpc::ClientContext context;
    context.AddMetadata("n_nodes_in_level", std::to_string(n_nodes_in_level));
    grpc::Status status = stub_->TriggerAggregate(&context, id, &ready);
    if (!status.ok()) {
        LOG(ERROR) << "TriggerAggregate rpc failed.";
        return false;
    } else if (!ready.ready()) {
        LOG(ERROR) << "Server has not received all histograms.";
        return false;
    } else {
        return true;
    }
}

void DistributedParty::GetBestInfo(vector<BestInfo> &bests) {
    fedtree::PID id;
    fedtree::BestInfo best;
    grpc::ClientContext context;
    auto t_start = timer.now();
    id.set_id(pid);
    std::unique_ptr<grpc::ClientReader<fedtree::BestInfo> > reader(stub_->GetBestInfo(&context, id));
    while (reader->Read(&best)) {
        bests.push_back({best.pid(), best.nid(), best.idx(), best.global_fid(), static_cast<float_type>(best.gain())});
    }

    grpc::Status status = reader->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += bests.size() * 24 * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "All nodes updated using best info.";
    } else {
        LOG(ERROR) << "GetBestInfo rpc failed.";
    }
}

void DistributedParty::SendNode(Tree::TreeNode &node_data) {
    fedtree::Node node;
    fedtree::PID id;
    grpc::ClientContext context;
    LOG(INFO)<<"communication Node start";
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    node.set_final_id(node_data.final_id);
    node.set_lch_index(node_data.lch_index);
    node.set_rch_index(node_data.rch_index);
    node.set_parent_index(node_data.parent_index);
    node.set_gain(node_data.gain);
    node.set_base_weight(node_data.base_weight);
    node.set_split_feature_id(node_data.split_feature_id);
    node.set_pid(node_data.pid);
    node.set_split_value(node_data.split_value);
    node.set_split_bid(node_data.split_bid);
    node.set_default_right(node_data.default_right);
    node.set_is_leaf(node_data.is_leaf);
    node.set_is_valid(node_data.is_valid);
    node.set_is_pruned(node_data.is_pruned);
    node.set_sum_gh_pair_g(node_data.sum_gh_pair.g);
    node.set_sum_gh_pair_h(node_data.sum_gh_pair.h);
    node.set_n_instances(node_data.n_instances);
    grpc::Status status = stub_->SendNode(&context, node, &id);
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication Node end";
    comm_time += used_time.count();
    comm_size += node.ByteSizeLong() * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "Node sent.";
    } else {
        LOG(ERROR) << "SendNodes rpc failed.";
    }
}

void DistributedParty::OrganizeNodes(fedtree::NodeArray &nodes, Tree::TreeNode &node_data){
    nodes.add_final_id(node_data.final_id);
    nodes.add_lch_index(node_data.lch_index);
    nodes.add_rch_index(node_data.rch_index);
    nodes.add_parent_index(node_data.parent_index);
    nodes.add_gain(node_data.gain);
    nodes.add_base_weight(node_data.base_weight);
    nodes.add_split_feature_id(node_data.split_feature_id);
    nodes.add_pid(node_data.pid);
    nodes.add_split_value(node_data.split_value);
    nodes.add_split_bid(node_data.split_bid);
    nodes.add_default_right(node_data.default_right);
    nodes.add_is_leaf(node_data.is_leaf);
    nodes.add_is_valid(node_data.is_valid);
    nodes.add_is_pruned(node_data.is_pruned);
    nodes.add_sum_gh_pair_g(node_data.sum_gh_pair.g);
    nodes.add_sum_gh_pair_h(node_data.sum_gh_pair.h);
    nodes.add_n_instances(node_data.n_instances);
}

void DistributedParty::SendNodes(fedtree::NodeArray &nodes) {
    fedtree::PID id;
    grpc::ClientContext context;
    LOG(INFO)<<"communication Nodes start";
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    grpc::Status status = stub_->SendNodes(&context, nodes, &id);
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication Nodes end";
    comm_time += used_time.count();
    comm_size += nodes.ByteSizeLong() * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "Node Enc sent.";
    } else {
        LOG(ERROR) << "SendNodes rpc failed.";
    }
}

void DistributedParty::OrganizeNodesEnc(fedtree::NodeEncArray &nodes, Tree::TreeNode &node_data){
    nodes.add_final_id(node_data.final_id);
    nodes.add_lch_index(node_data.lch_index);
    nodes.add_rch_index(node_data.rch_index);
    nodes.add_parent_index(node_data.parent_index);
    nodes.add_gain(node_data.gain);
    nodes.add_base_weight(node_data.base_weight);
    nodes.add_split_feature_id(node_data.split_feature_id);
    nodes.add_pid(node_data.pid);
    nodes.add_split_value(node_data.split_value);
    nodes.add_split_bid(node_data.split_bid);
    nodes.add_default_right(node_data.default_right);
    nodes.add_is_leaf(node_data.is_leaf);
    nodes.add_is_valid(node_data.is_valid);
    nodes.add_is_pruned(node_data.is_pruned);
    assert(node_data.sum_gh_pair.encrypted);
    stringstream stream;
    stream<<node_data.sum_gh_pair.g_enc;
    nodes.add_sum_gh_pair_g_enc(stream.str());
    stream.clear();
    stream.str("");
    stream<<node_data.sum_gh_pair.h_enc;
    nodes.add_sum_gh_pair_h_enc(stream.str());
    stream.clear();
    stream.str("");
    nodes.add_n_instances(node_data.n_instances);
}

void DistributedParty::SendNodesEnc(fedtree::NodeEncArray &nodes) {
    fedtree::PID id;
    grpc::ClientContext context;
    LOG(INFO)<<"communication NodeEnc start";
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    grpc::Status status = stub_->SendNodesEnc(&context, nodes, &id);
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication NodeEnc end";
    comm_time += used_time.count();
    comm_size += nodes.ByteSizeLong() * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "Node Enc sent.";
    } else {
        LOG(ERROR) << "SendNodes rpc failed.";
    }
}

void DistributedParty::SendNodeEnc(Tree::TreeNode &node_data) {
    fedtree::NodeEnc node;
    fedtree::PID id;
    grpc::ClientContext context;
    LOG(INFO)<<"communication NodeEnc start";
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    node.set_final_id(node_data.final_id);
    node.set_lch_index(node_data.lch_index);
    node.set_rch_index(node_data.rch_index);
    node.set_parent_index(node_data.parent_index);
    node.set_gain(node_data.gain);
    node.set_base_weight(node_data.base_weight);
    node.set_split_feature_id(node_data.split_feature_id);
    node.set_pid(node_data.pid);
    node.set_split_value(node_data.split_value);
    node.set_split_bid(node_data.split_bid);
    node.set_default_right(node_data.default_right);
    node.set_is_leaf(node_data.is_leaf);
    node.set_is_valid(node_data.is_valid);
    node.set_is_pruned(node_data.is_pruned);
    if (node_data.sum_gh_pair.encrypted) {
        stringstream stream;
        stream<<node_data.sum_gh_pair.g_enc;
        node.set_sum_gh_pair_g_enc(stream.str());
        stream.clear();
        stream.str("");
        stream<<node_data.sum_gh_pair.h_enc;
        node.set_sum_gh_pair_h_enc(stream.str());
        stream.clear();
        stream.str("");
    }
    node.set_n_instances(node_data.n_instances);
    node.set_is_enc(node_data.sum_gh_pair.encrypted);
    grpc::Status status = stub_->SendNodeEnc(&context, node, &id);
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication NodeEnc end";
    comm_time += used_time.count();
    comm_size += node.ByteSizeLong() * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "Node Enc sent.";
    } else {
        LOG(ERROR) << "SendNodes rpc failed.";
    }
}

void DistributedParty::SendIns2NodeID(SyncArray<int> &ins2node_id, int nid, int l) {
    fedtree::PID id;
    grpc::ClientContext context;
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    context.AddMetadata("l", std::to_string(l));
    std::unique_ptr<grpc::ClientWriter<fedtree::Ins2NodeID> > writer(
            stub_->SendIns2NodeID(&context, &id));

    auto ins2node_id_data = ins2node_id.host_data();
    long total_size = 0;
    for (int i = 0; i < ins2node_id.size(); ++i) {
        if (ins2node_id_data[i] >= 2 * nid + 1 && ins2node_id_data[i] <= 2 * nid + 2) {
            fedtree::Ins2NodeID i2n;
            i2n.set_iid(i);
            i2n.set_nid(ins2node_id_data[i]);
            total_size += 1;
            if (!writer->Write(i2n))
                break;
        }
    }
    writer->WritesDone();
    grpc::Status status = writer->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += total_size * 8e-6;
    if (status.ok()) {
        LOG(DEBUG) << "ins2node_id of the current node sent.";
    } else {
        LOG(ERROR) << "SendIns2NodeID rpc failed.";
    }
}

void DistributedParty::GetNodes(int l) {
    fedtree::PID id;
    fedtree::Node node;
    grpc::ClientContext context;
    LOG(INFO)<<"communication Node start";
    auto t_start = timer.now();
    context.AddMetadata("l", std::to_string(l));
    id.set_id(pid);
    std::unique_ptr<grpc::ClientReader<fedtree::Node> > reader(stub_->GetNodes(&context, id));
    int i = 0;
    while (reader->Read(&node)) {
        i++;
        int nid = node.final_id();
        auto nodes_data = booster.fbuilder->trees.nodes.host_data();
        nodes_data[nid].final_id = node.final_id();
        nodes_data[nid].lch_index = node.lch_index();
        nodes_data[nid].rch_index = node.rch_index();
        nodes_data[nid].parent_index = node.parent_index();
        nodes_data[nid].gain = node.gain();
        nodes_data[nid].base_weight = node.base_weight();
        nodes_data[nid].split_feature_id = node.split_feature_id();
        nodes_data[nid].pid = node.pid();
        nodes_data[nid].split_value = node.split_value();
        nodes_data[nid].split_bid = node.split_bid();
        nodes_data[nid].default_right = node.default_right();
        nodes_data[nid].is_leaf = node.is_leaf();
        nodes_data[nid].is_valid = node.is_valid();
        nodes_data[nid].is_pruned = node.is_pruned();
        nodes_data[nid].sum_gh_pair.encrypted = false;
        nodes_data[nid].sum_gh_pair.g = node.sum_gh_pair_g();
        nodes_data[nid].sum_gh_pair.h = node.sum_gh_pair_h();
        nodes_data[nid].n_instances = node.n_instances();
    }
    grpc::Status status = reader->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication Node end";
    comm_time += used_time.count();
    comm_size += i * node.ByteSizeLong() * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "All nodes received." << i;
    } else {
        LOG(ERROR) << "GetNodes rpc failed.";
    }
}

void DistributedParty::GetIns2NodeID() {
    fedtree::PID id;
    fedtree::Ins2NodeID i2n;
    grpc::ClientContext context;
    auto t_start = timer.now();
    id.set_id(pid);
    LOG(DEBUG) << "Receiving ins2node_id from the server.";// << std::endl;
    auto ins2node_id_data = booster.fbuilder->ins2node_id.host_data();
    std::unique_ptr<grpc::ClientReader<fedtree::Ins2NodeID> > reader(stub_->GetIns2NodeID(&context, id));
    int total_size = 0;
    while (reader->Read(&i2n)) {
        int iid = i2n.iid();
        int nid = i2n.nid();
        ins2node_id_data[iid] = nid;
        total_size++;
    }
    grpc::Status status = reader->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += total_size * 8e-6;
    if (status.ok()) {
        LOG(DEBUG) << "All ins2node_id received.";
    } else {
        LOG(ERROR) << "GetIns2NodeID rpc failed.";
    }
}

bool DistributedParty::CheckIfContinue(bool cont) {
    fedtree::PID id;
    fedtree::Ready ready;
    grpc::ClientContext context;
    auto t_start = timer.now();
    context.AddMetadata("cont", std::to_string(cont));
    id.set_id(pid);
    grpc::Status status = stub_->CheckIfContinue(&context, id, &ready);
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += id.ByteSizeLong() * 1e-6;
    if (!status.ok()) {
        LOG(ERROR) << "CheckIfContinue rpc failed.";
        return false;
    } else if (!ready.ready()) {
        LOG(DEBUG) << "No further splits, stop.";
        return false;
    } else {
        return true;
    }
}

void DistributedParty::TriggerPrune(int t) {
    fedtree::PID id;
    fedtree::Ready ready;
    grpc::ClientContext context;
    context.AddMetadata("t", std::to_string(t));
    grpc::Status status = stub_->TriggerPrune(&context, id, &ready);
    if (status.ok()) {
        LOG(DEBUG) << "Triggered the server to prune.";
    } else {
        LOG(ERROR) << "TriggerPrune rpc failed.";
    }
}

void DistributedParty::TriggerPrintScore() {
    fedtree::PID id;
    fedtree::Ready ready;
    grpc::ClientContext context;
    grpc::Status status = stub_->TriggerPrintScore(&context, id, &ready);
    if (status.ok()) {
        LOG(DEBUG) << "Triggered the server to print score.";
    } else {
        LOG(ERROR) << "TriggerPrintScore rpc failed.";
    }
}

void DistributedParty::SendRange(const vector<vector<float_type>>& ranges) {
    fedtree::PID id;
    grpc::ClientContext context;
    LOG(INFO)<<"communication Range start";
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    std::unique_ptr<grpc::ClientWriter<fedtree::GHPair>> writer(stub_->SendRange(&context, &id));
    for (int i = 0; i < ranges.size(); i++) {
        fedtree::GHPair range;
        range.set_g(ranges[i][0]);
        range.set_h(ranges[i][1]);
        if (!writer->Write(range)) {
            break;
        }
    }
    writer->WritesDone();
    grpc::Status status = writer->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication Range end";
    comm_time += used_time.count();
    comm_size += ranges.size() * 16e-6;
    if (status.ok()) {
        LOG(DEBUG) << "All feature range sent.";
    }
    else {
        LOG(ERROR) << "SendRange rpc failed.";
    }
}

void DistributedParty::TriggerCut(int n_bins) {
    fedtree::PID id;
    fedtree::Ready ready;
    grpc::ClientContext context;
    id.set_id(n_bins);
    grpc::Status status = stub_->TriggerCut(&context, id, &ready);
    if (status.ok()) {
        LOG(DEBUG) << "Triggered the server to cut.";
    } else {
        LOG(ERROR) << "TriggerCut rpc failed.";
    }
}

void DistributedParty::GetRangeAndSet(int n_bins) {
    grpc::ClientContext context;
    fedtree::PID id;
    LOG(INFO)<<"communication RangeSet start";
    auto t_start = timer.now();
    id.set_id(pid);
    std::unique_ptr<grpc::ClientReader<fedtree::GHPair>> reader(stub_->GetRange(&context, id));
    fedtree::GHPair range;
    vector<vector<float_type>> feature_range;
    while(reader->Read(&range)) {
        feature_range.push_back({range.g(), range.h()});
    }
    grpc::Status status = reader->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication RangeSet end";
    comm_time += used_time.count();
    comm_size += feature_range.size() * 16e-6;
    if (status.ok()) {
        LOG(DEBUG) << "All range received.";
    } else {
        LOG(ERROR) << "GetRange rpc failed.";
    }
    booster.fbuilder->cut.get_cut_points_by_feature_range(feature_range, n_bins);
    booster.fbuilder->get_bin_ids();
}

void DistributedParty::SendGH(GHPair party_gh) {
    fedtree::GHPair pair;
    fedtree::PID id;
    grpc::ClientContext context;
    LOG(INFO)<<"communication GH start";
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    pair.set_g(party_gh.g);
    pair.set_h(party_gh.h);
    grpc::Status status = stub_->SendGH(&context, pair, &id);
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication GH end";
    comm_time += used_time.count();
    comm_size += pair.ByteSizeLong() * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "party_gh sent.";
    }
    else {
        LOG(ERROR) << "party_gh rpc failed";
    }
}

void DistributedParty::SendDHPubKey() {
    fedtree::DHPublicKey pk;
    fedtree::PID id;
    grpc::ClientContext context;
    LOG(INFO)<<"communication DHPublicKey start";
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    stringstream stream;
    stream<<dh.public_key;
    pk.set_pk(stream.str());
    stream.clear();
    grpc::Status status = stub_->SendDHPubKey(&context, pk, &id);
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication DHPublicKey end";
    comm_time += used_time.count();
    comm_size += pk.ByteSizeLong() * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "DHPublicKey sent.";
    }
    else {
        LOG(ERROR) << "DHPublicKey rpc failed";
    }
}

void DistributedParty::GetDHPubKey() {
    grpc::ClientContext context;
    fedtree::PID id;
    fedtree::DHPublicKeys pk;
    LOG(INFO)<<"communication PublicKey start";
    auto t_start = timer.now();
    id.set_id(pid);
    std::unique_ptr<grpc::ClientReader<fedtree::DHPublicKeys> > reader(stub_->GetDHPubKeys(&context, id));
    dh.other_public_keys.SetLength(n_parties);
    reader->Read(&pk);
    for(int i = 0; i < pk.pk_size(); i++) {
        dh.other_public_keys[i] = NTL::to_ZZ(pk.pk(i).c_str());
    }
    grpc::Status status = reader->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += pk.ByteSizeLong() * 1e-6;
    LOG(INFO)<<"communication PublicKey end";
    if (status.ok()) {
        LOG(INFO) << "Get DHPubKey from server";
    }
    else {
        LOG(ERROR) << "GetDHPubKey rpc failed";
    }
}

void DistributedParty::SendNoises(){
    fedtree::SANoises san;
    fedtree::PID id;
    grpc::ClientContext context;
    LOG(INFO)<<"communication Noises start";
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    stringstream stream;
    for(int i = 0; i < dh.encrypted_noises.length(); i++){
        stream<<dh.encrypted_noises[i];
        san.add_noises(stream.str());
        stream.clear();
        stream.str("");
    }
    grpc::Status status = stub_->SendNoises(&context, san, &id);
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication Noises end";
    comm_time += used_time.count();
    comm_size += san.ByteSizeLong() * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "Noises sent.";
    }
    else {
        LOG(ERROR) << "Noises rpc failed";
    }
}

void DistributedParty::GetNoises() {
    grpc::ClientContext context;
    fedtree::PID id;
    fedtree::SANoises san;
    LOG(INFO)<<"communication Noises start";
    auto t_start = timer.now();
    id.set_id(pid);
    std::unique_ptr<grpc::ClientReader<fedtree::SANoises> > reader(stub_->GetNoises(&context, id));
    dh.received_encrypted_noises.SetLength(n_parties);
    reader->Read(&san);
    for(int i = 0; i < san.noises_size(); i++){
        dh.received_encrypted_noises[i] = NTL::to_ZZ(san.noises(i).c_str());
    }
//    while (reader->Read(&san)) {
//        dh.received_encrypted_noises[pid] = NTL::to_ZZ(san.noises().c_str());
//        pid++;
//    }
    grpc::Status status = reader->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += san.ByteSizeLong() * 1e-6;
    LOG(INFO)<<"communication Noises end";
    if (status.ok()) {
        LOG(INFO) << "GetNoises from server";
    }
    else {
        LOG(ERROR) << "GetNoises rpc failed";
    }
}

void DistributedParty::SendCutPoints(){
    fedtree::CutPoints cp;
    fedtree::PID id;
    grpc::ClientContext context;
    LOG(INFO)<<"communication CutPoints start";
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    stringstream stream;
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
    grpc::Status status = stub_->SendCutPoints(&context, cp, &id);
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication CutPoints end";
    comm_time += used_time.count();
    comm_size += cp.ByteSizeLong() * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "CutPoints sent.";
    }
    else {
        LOG(ERROR) << "CutPoints rpc failed";
    }
}

void DistributedParty::GetCutPoints() {
    grpc::ClientContext context;
    fedtree::PID id;
    fedtree::CutPoints cp;
    LOG(INFO)<<"communication CutPoints start";
    auto t_start = timer.now();
    id.set_id(pid);
    std::unique_ptr<grpc::ClientReader<fedtree::CutPoints> > reader(stub_->GetCutPoints(&context, id));
    reader->Read(&cp);

    int cut_points_val_size = cp.cut_points_val_size();
    int cut_col_ptr_size = cp.cut_col_ptr_size();
    booster.fbuilder->cut.cut_points_val.resize(cut_points_val_size);
    booster.fbuilder->cut.cut_col_ptr.resize(cut_col_ptr_size);
    booster.fbuilder->cut.cut_fid.resize(cut_points_val_size);

    auto cut_val_data = booster.fbuilder->cut.cut_points_val.host_data();
    auto cut_col_ptr_data = booster.fbuilder->cut.cut_col_ptr.host_data();
    auto cut_fid_data = booster.fbuilder->cut.cut_fid.host_data();
    for(int i = 0; i < cut_points_val_size; i++){
        cut_val_data[i] = cp.cut_points_val(i);
        cut_fid_data[i] = cp.cut_fid(i);
    }
    for(int i = 0; i < cut_col_ptr_size; i++){
        cut_col_ptr_data[i] = cp.cut_col_ptr(i);
    }
//    while (reader->Read(&san)) {
//        dh.received_encrypted_noises[pid] = NTL::to_ZZ(san.noises().c_str());
//        pid++;
//    }
    grpc::Status status = reader->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += cp.ByteSizeLong() * 1e-6;
    LOG(INFO)<<"communication CutPoints end";
    booster.fbuilder->get_bin_ids();
    if (status.ok()) {
        LOG(INFO) << "GetCutPoints from server";
    }
    else {
        LOG(ERROR) << "GetCutPoints rpc failed";
    }
}

void DistributedParty::TriggerBuildUsingGH(int k) {
    grpc::ClientContext context;
    context.AddMetadata("k", std::to_string(k));
    fedtree::PID id;
    fedtree::Ready ready;
    grpc::Status status = stub_->TriggerBuildUsingGH(&context, id, &ready);
    if (status.ok()) {
        LOG(DEBUG) << "Triggered the server to build use gh.";

    }
    else {
        LOG(ERROR) << "TriggerBuildUsingGH failed";
    }
}

void DistributedParty::TriggerCalcTree(int l) {
    grpc::ClientContext context;
    context.AddMetadata("l", std::to_string(l));
    fedtree::PID id;
    fedtree::Ready ready;
    grpc::Status status = stub_->TriggerCalcTree(&context, id, &ready);
    if (status.ok()) {
        LOG(DEBUG) << "Triggerd the server to calc tree.";
    }
    else {
        LOG(ERROR) << "TriggerCalcTree failed.";
    }
}

void DistributedParty::GetRootNode() {
    grpc::ClientContext context;
    fedtree::PID id;
    fedtree::Node node;
    LOG(INFO)<<"communication RootNode start";
    auto t_start = timer.now();
    grpc::Status status = stub_->GetRootNode(&context, id, &node);
    auto t_end = timer.now();
    LOG(INFO)<<"communication RootNode end";
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += node.ByteSizeLong() * 1e-6;
    auto & root = booster.fbuilder->trees.nodes.host_data()[0];
    root.final_id = node.final_id();
    root.lch_index = node.lch_index();
    root.rch_index = node.rch_index();
    root.parent_index = node.parent_index();
    root.gain = node.gain();
    root.base_weight = node.base_weight();
    root.split_feature_id = node.split_feature_id();
    root.pid = node.pid();
    root.split_value = node.split_value();
    root.split_bid = node.split_bid();
    root.default_right = node.default_right();
    root.is_leaf = node.is_leaf();
    root.is_valid = node.is_valid();
    root.is_pruned = node.is_pruned();
    root.sum_gh_pair.g = node.sum_gh_pair_g();
    root.sum_gh_pair.h = node.sum_gh_pair_h();
    root.n_instances = node.n_instances();
    if (status.ok()) {
        LOG(DEBUG) << "RootNode received." << std::endl;
    } else {
        LOG(ERROR) << "GetRootNode rpc failed." << std::endl;
    }
}

void DistributedParty::GetSplitPoints() {
    grpc::ClientContext context;
    fedtree::PID id;
    LOG(INFO)<<"communication SplitPoints start";
    auto t_start = timer.now();
    id.set_id(pid);
    std::unique_ptr<grpc::ClientReader<fedtree::SplitPoint>> reader(stub_->GetSplitPoints(&context, id));
    // TODO
    vector<SplitPoint> sp_points;
    fedtree::SplitPoint sp_recv;
    {
        TIMED_SCOPE(timerObj, "GetSPsTime");
    while(reader->Read(&sp_recv)) {
        SplitPoint sp_point;
        sp_point.gain = sp_recv.gain();
        sp_point.fea_missing_gh = {sp_recv.fea_missing_g(), sp_recv.fea_missing_h()};
        sp_point.rch_sum_gh = {sp_recv.rch_sum_g(), sp_recv.rch_sum_h()};
        sp_point.default_right = sp_recv.default_right();
        sp_point.nid = sp_recv.nid();
        sp_point.split_fea_id = sp_recv.split_fea_id();
        sp_point.fval = sp_recv.fval();
        sp_point.split_bid = sp_recv.split_bid();
        sp_point.no_split_value_update = sp_recv.no_split_value_update();
        sp_points.push_back(sp_point);
    }

    grpc::Status status = reader->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication SplitPoints end";
    comm_time += used_time.count();
    comm_size += sp_recv.ByteSizeLong() * 1e-6;
    if (status.ok()) {
        LOG(DEBUG) << "SplitPoints received.";
    } else {
        LOG(ERROR) << "GetSplitPoints rpc failed.";
    }
    }
    booster.fbuilder->sp.resize(sp_points.size());
    auto sp_data = booster.fbuilder->sp.host_data();
    for (int i = 0; i < sp_points.size(); i++) {
        sp_data[i] = sp_points[i];
    }

}

bool DistributedParty::HCheckIfContinue() {
    fedtree::PID id;
    fedtree::Ready ready;
    grpc::ClientContext context;
    LOG(INFO)<<"communication IfContinue start";
    auto t_start = timer.now();
    context.AddMetadata("cont", std::to_string(booster.fbuilder->has_split));
    grpc::Status status = stub_->HCheckIfContinue(&context, id, &ready);
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication IfContinue end";
    comm_time += used_time.count();
    comm_size += 16e-6;
    if (!status.ok()) {
        LOG(ERROR) << "HCheckIfContinue rpc failed.";
        return false;
    } else if (!ready.ready()) {
        LOG(DEBUG) << "No further splits, stop.";
        return false;
    } else {
        return true;
    }
}

float DistributedParty::GetAvgScore(float score) {
    grpc::ClientContext context;
    fedtree::Score s;
    fedtree::Score avg;
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    s.set_content(score);
    grpc::Status status = stub_->ScoreReduce(&context, s, &avg);
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += 12e-6;
    if (status.ok()) {
        LOG(DEBUG) << "Average score received";
    }
    else {
        LOG(ERROR) << "ScoreReduce rpc failed";
    }
    return avg.content();
}

void DistributedParty::TriggerHomoInit() {
    grpc::ClientContext context;
    fedtree::PID id;
    fedtree::Ready ready;
    grpc::Status status = stub_->TriggerHomoInit(&context, id, &ready);
    if (status.ok()) {
        LOG(DEBUG) << "Trigger Server to homo init";
    }
    else {
        LOG(ERROR) << "TriggerHomoInit rpc failed";
    }
}

void DistributedParty::TriggerSAInit() {
    grpc::ClientContext context;
    fedtree::PID id;
    fedtree::Ready ready;
    grpc::Status status = stub_->TriggerSAInit(&context, id, &ready);
    if (status.ok()) {
        LOG(DEBUG) << "Trigger Server to SA init";
    }
    else {
        LOG(ERROR) << "TriggerSAInit rpc failed";
    }
}


void DistributedParty::GetPaillier() {
    grpc::ClientContext context;
    fedtree::PID id;
    fedtree::Paillier pubkey;
    LOG(INFO)<<"communication PublicKey start";
    auto t_start = timer.now();
    id.set_id(pid);

    grpc::Status status = stub_->GetPaillier(&context, id, &pubkey);
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += (pubkey.ByteSizeLong()+4) * 1e-6;
    LOG(INFO)<<"communication PublicKey end";
    if (status.ok()) {
        paillier.modulus = NTL::to_ZZ(pubkey.modulus().c_str());
        paillier.generator = NTL::to_ZZ(pubkey.generator().c_str());
//        LOG(INFO) << "Get public key from server";
    }
    else {
        LOG(ERROR) << "GetPaillier rpc failed";
    }
}

void DistributedParty::SendHistogramsEnc(const SyncArray<GHPair> &hist, int type) {
    fedtree::PID id;
    grpc::ClientContext context;
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    context.AddMetadata("type", std::to_string(type));
    
    TIMED_SCOPE(timerObj, "SendHistogramsEncTime");
    std::unique_ptr<grpc::ClientWriter<fedtree::GHPairEnc> > writer(
            stub_->SendHistogramsEnc(&context, &id));
    
    auto hist_data = hist.host_data();
    stringstream stream;
    for (int i = 0; i < hist.size(); ++i) {
        fedtree::GHPairEnc gh;
        stream << hist_data[i].g_enc;
        gh.set_g_enc(stream.str());
        stream.clear();
        stream.str("");
        stream << hist_data[i].h_enc;
        gh.set_h_enc(stream.str());
        stream.clear();
        stream.str("");
        if (!writer->Write(gh)) {
            break;
        }
    }
    writer->WritesDone();
    
    grpc::Status status = writer->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    comm_time += used_time.count();
    comm_size += hist.size() * 16e-6 + 4e-6;
    if (status.ok()) {
        LOG(DEBUG) << "All Encrypted" << type << " sent.";
    } else {
        LOG(ERROR) << "SendHistogramsEnc rpc failed.";
    }

}

void DistributedParty::SendHistogramBatches(const SyncArray<GHPair> &hist, int type) {
    fedtree::PID id;
    grpc::ClientContext context;
    LOG(INFO)<<"communication HistBatch start";
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    context.AddMetadata("type", std::to_string(type));
    TIMED_SCOPE(timerObj, "SendHistogramBatchesTime");
    auto hist_data = hist.host_data();
    const int BATCH_SIZE = 200000;
    vector<fedtree::GHBatch> tmp;
    int len = hist.size();
    tmp.resize((len + BATCH_SIZE - 1)/BATCH_SIZE);
    
    for (int beg = 0; beg < len; beg += BATCH_SIZE) {
        int end = min(len, beg+BATCH_SIZE);
        for (int i = beg; i < end; i++) {
            tmp[beg/BATCH_SIZE].add_g(hist_data[i].g);
            tmp[beg/BATCH_SIZE].add_h(hist_data[i].h);
        }
    }
    std::unique_ptr<grpc::ClientWriter<fedtree::GHBatch> > writer(
            stub_->SendHistogramBatches(&context, &id));
    for (const auto &e: tmp) {
        if (!writer->Write(e)) {
            break;
        }
    }
    
    writer->WritesDone();
    grpc::Status status = writer->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication HistBatch end";
    comm_time += used_time.count();
    comm_size += hist.size() * 16e-6 + (len + BATCH_SIZE - 1)/BATCH_SIZE*4e-6;
    if (status.ok()) {
        LOG(DEBUG) << "All " << type << " sent.";
    } else {
        LOG(ERROR) << "SendHistogramBatches rpc failed.";
    }
}

void DistributedParty::SendHistFidBatches(const SyncArray<int> &hist) {
    fedtree::PID id;
    grpc::ClientContext context;
    LOG(INFO)<<"communication HistFidBatch start";
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
//    TIMED_SCOPE(timerObj, "SendHistFidBatchesTime");
    std::unique_ptr<grpc::ClientWriter<fedtree::FIDBatch> > writer(
            stub_->SendHistFidBatches(&context, &id));

    auto hist_data = hist.host_data();
    const int BATCH_SIZE = 200000;
    
    vector<fedtree::FIDBatch> tmp;
    for (int beg = 0; beg < hist.size(); beg += BATCH_SIZE) {
        int end = min((int)hist.size(), beg+BATCH_SIZE);
        tmp.emplace_back();
        for (int i = beg; i < end; i++) {
            tmp.back().add_id(hist_data[i]);
        }
    }
    for (const auto &e: tmp) {
        if (!writer->Write(e)) {
            break;
        }
    }
    
    writer->WritesDone();
    grpc::Status status = writer->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication HistFidBatch end";
    comm_time += used_time.count();
    comm_size += hist.size() * 4e-6 + tmp.size() * 4e-6;
    if (status.ok()) {
        LOG(DEBUG) << "All HistFid Batches sent";
    } else {
        LOG(ERROR) << "SendHistFIDBatches rpc failed.";
    }
}
void DistributedParty::GetIns2NodeIDBatches() {
    fedtree::PID id;
    fedtree::Ins2NodeIDBatch i2n;
    grpc::ClientContext context;
    LOG(INFO)<<"communication Ins2NodeID start";
    auto t_start = timer.now();
    id.set_id(pid);
    LOG(DEBUG) << "Receiving ins2node_id from the server.";// << std::endl;

    auto ins2node_id_data = booster.fbuilder->ins2node_id.host_data();
    std::unique_ptr<grpc::ClientReader<fedtree::Ins2NodeIDBatch> > reader(stub_->GetIns2NodeIDBatches(&context, id));
    int total_size = 0;
    while (reader->Read(&i2n)) {
        int len = i2n.iid_size();
        for (int i = 0; i < len; i++) {
            int iid = i2n.iid(i);
            int nid = i2n.nid(i);
            ins2node_id_data[iid] = nid;
        }
        total_size+=len;
    }
    grpc::Status status = reader->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication Ins2NodeID end";
    comm_time += used_time.count();
    comm_size += total_size * 8e-6;
    if (status.ok()) {
        LOG(DEBUG) << "All ins2node_id received.";
    } else {
        LOG(ERROR) << "GetIns2NodeIDBatches rpc failed.";
    }
}
void DistributedParty::SendIns2NodeIDBatches(SyncArray<int> &ins2node_id, int nid, int l) {
    fedtree::PID id;
    grpc::ClientContext context;
    LOG(INFO)<<"communication Ins2NodeID start";
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    context.AddMetadata("l", std::to_string(l));
    std::unique_ptr<grpc::ClientWriter<fedtree::Ins2NodeIDBatch> > writer(
            stub_->SendIns2NodeIDBatches(&context, &id));

    auto ins2node_id_data = ins2node_id.host_data();
    fedtree::Ins2NodeIDBatch i2n;
    const int BATCH_SIZE = 200000;

    int total_size = 0;
    for (int i = 0; i < ins2node_id.size(); ++i) {
        if (ins2node_id_data[i] >= 2 * nid + 1 && ins2node_id_data[i] <= 2 * nid + 2) {
            // fedtree::Ins2NodeID i2n;
            i2n.add_iid(i);
            i2n.add_nid(ins2node_id_data[i]);
            total_size ++;
            if (i2n.iid_size() >= 200000){
                if (!writer->Write(i2n))
                    break;
                i2n.clear_iid();
                i2n.clear_nid();
        }   }
    }
    if (i2n.iid_size() > 0) {
        writer->Write(i2n);
    }

    writer->WritesDone();
    grpc::Status status = writer->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication Ins2NodeID end";
    comm_time += used_time.count();
    comm_size += total_size * 8e-6;
    if (status.ok()) {
        LOG(DEBUG) << "ins2node_id of the current node sent.";
    } else {
        LOG(ERROR) << "SendIns2NodeIDBatches rpc failed.";
    }
}

void DistributedParty::GetGradientBatches() {
    fedtree::PID id;
    fedtree::GHBatch gh;
    fedtree::GHBatch tot;
    grpc::ClientContext context;
    LOG(INFO)<<"communication GradientBatches start";
    auto t_start = timer.now();
    id.set_id(pid);
    LOG(DEBUG) << "Receiving gradients from the server.";

    std::unique_ptr<grpc::ClientReader<fedtree::GHBatch> > reader(stub_->GetGradientBatches(&context, id));

    auto booster_gradients_data = booster.gradients.host_data();
    
    while (reader->Read(&gh)) {
        tot.MergeFrom(gh);
    }
    grpc::Status status = reader->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication GradientBatches end";
    comm_time += used_time.count();
    int len = booster.gradients.size();
    comm_size += len * 16e-6;

    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        booster_gradients_data[i] = {static_cast<float_type>(tot.g(i)), static_cast<float_type>(tot.h(i))};
    }
    if (status.ok()) {
        LOG(DEBUG) << "All gradients received.";
    } else {
        LOG(ERROR) << "GetGradients rpc failed.";
    }
}

void DistributedParty::GetGradientBatchesEnc() {
    fedtree::PID id;
    fedtree::GHEncBatch gh;
    fedtree::GHEncBatch tot;
    grpc::ClientContext context;
    LOG(INFO)<<"communication GradientBatchesEnc start";
    auto t_start = timer.now();
    id.set_id(pid);
    LOG(DEBUG) << "Receiving gradients from the server.";

    std::unique_ptr<grpc::ClientReader<fedtree::GHEncBatch>> reader(stub_->GetGradientBatchesEnc(&context, id));

    auto booster_gradients_data = booster.gradients.host_data();
    
    while (reader->Read(&gh)) {
        tot.MergeFrom(gh);
    }
    grpc::Status status = reader->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication GradientBatchesEnc end";
    comm_time += used_time.count();
    int len = booster.gradients.size();
    comm_size += len*16e-6;
    assert(len == tot.g_enc_size());
    assert(len == tot.h_enc_size());
    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
        booster_gradients_data[i].encrypted = true;
        booster_gradients_data[i].paillier = paillier;
        booster_gradients_data[i].g_enc = NTL::to_ZZ(tot.g_enc(i).c_str());
        booster_gradients_data[i].h_enc = NTL::to_ZZ(tot.h_enc(i).c_str());
        
    }
    if (status.ok()) {
        LOG(DEBUG) << "All gradients received.";
    } else {
        LOG(ERROR) << "GetGradients rpc failed.";
    }
}


void DistributedParty::SendHistogramBatchesEnc(const SyncArray<GHPair> &hist, int type) {
    fedtree::PID id;
    grpc::ClientContext context;
    auto hist_data = hist.host_data();
    vector<fedtree::GHEncBatch> tmp;
    {
    TIMED_SCOPE(timerObj, "encodeTime");
    
    const int BATCH_SIZE = 5000;
    
    int len = hist.size();
    tmp.resize((len+BATCH_SIZE-1)/BATCH_SIZE);
    #pragma omp parallel for
    for (int beg = 0; beg < len; beg += BATCH_SIZE) {
        stringstream stream;
        int end = min((int)hist.size(), beg+BATCH_SIZE);
        for (int i = beg; i < end; i++) {
            stream << hist_data[i].g_enc;
            tmp[beg/BATCH_SIZE].add_g_enc(stream.str());
            stream.clear();
            stream.str("");
            stream << hist_data[i].h_enc;
            tmp[beg/BATCH_SIZE].add_h_enc(stream.str());
            stream.clear();
            stream.str("");
        }
    }
    }
    LOG(INFO)<<"communication HistBatchEnc start";
    auto t_start = timer.now();
    context.AddMetadata("pid", std::to_string(pid));
    context.AddMetadata("type", std::to_string(type));
    std::unique_ptr<grpc::ClientWriter<fedtree::GHEncBatch> > writer(
            stub_->SendHistogramBatchesEnc(&context, &id));
    TIMED_SCOPE(timerObj, "SendHistogramBatchesEnc");
    for (const auto &e: tmp) {
        if (!writer->Write(e)) {
            break;
        }
    }
    writer->WritesDone();
    
    grpc::Status status = writer->Finish();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO)<<"communication HistBatchEnc end";
    comm_time += used_time.count();
    comm_size += hist.size()*16e-6;
    if (status.ok()) {
        LOG(DEBUG) << "All " << type << " sent.";
    } else {
        LOG(ERROR) << "SendHistogramBatchesEnc rpc failed.";
    }
}

void DistributedParty::StopServer(float tot_time) {
    fedtree::PID id;
    grpc::ClientContext context;
    fedtree::Score resp;
    id.set_id(pid);
    context.AddMetadata("tot", std::to_string(tot_time));
    context.AddMetadata("comm", std::to_string(comm_time));
    context.AddMetadata("enc", std::to_string(enc_time));
    grpc::Status status = stub_->StopServer(&context, id, &resp);
    LOG(INFO) << "communication time: " << comm_time << "s";
    LOG(INFO) << "wait time: " << resp.content() << "s";
    LOG(INFO) << "real communication: " << comm_time - resp.content() << "s";
    LOG(INFO) << "communication size: " << comm_size << "MB";
    if (status.ok()) {
        LOG(DEBUG) << "StopServer rpc success.";
    } else {
        LOG(ERROR) << "StopServer rpc failed.";
    }
}

void DistributedParty::BeginBarrier() {
    fedtree::PID id;
    grpc::ClientContext context;
    fedtree::Ready resp;
    id.set_id(pid);
    grpc::Status status = stub_->BeginBarrier(&context, id, &resp);
    if (status.ok()) {
        LOG(INFO) << "BeginBarrier rpc success.";
    } else {
        LOG(ERROR) << "BeginBarrier rpc failed.";
    }
} 

void distributed_vertical_train(DistributedParty& party, FLParam &fl_param) {
    if (fl_param.privacy_tech == "he") {
        if (party.pid == 0) {
            party.TriggerHomoInit();
        }
        if (!party.has_label)
            party.GetPaillier();
    }
    GBDTParam &param = fl_param.gbdt_param;
//    party.SendDatasetInfo(party.booster.fbuilder->cut.cut_points_val.size(), party.dataset.n_features());
    for (int round = 0; round < param.n_trees; round++) {
        LOG(INFO) << "training round " << round << " start";
        vector<Tree> trees(param.tree_per_rounds);
        if (party.pid == 0)
            party.TriggerUpdateGradients();
        if (fl_param.privacy_tech == "he" && !party.has_label) {
            party.GetGradientBatchesEnc();
        }
        else {
            if(party.has_label)
                party.booster.update_gradients();
            else
                party.GetGradientBatches();
        }
        for (int t = 0; t < param.tree_per_rounds; t++) {
            Tree &tree = trees[t];
            party.booster.fbuilder->build_init(party.booster.gradients, t);
            if (party.pid == 0)
                party.TriggerBuildInit(t);
            for (int l = 0; l < param.depth; l++) {
                int n_nodes_in_level = 1 << l;
                int n_bins = party.booster.fbuilder->cut.cut_points_val.size();
                int n_max_nodes = 2 << param.depth;
                int n_max_splits = n_max_nodes * n_bins;
                int n_column = party.dataset.n_features();
                int n_partition = n_column * n_nodes_in_level;
                auto cut_fid_data = party.booster.fbuilder->cut.cut_fid.host_data();
                SyncArray<int> hist_fid(n_nodes_in_level * n_bins);
                auto hist_fid_data = hist_fid.host_data();
                for (int i = 0; i < hist_fid.size(); i++) {
                    hist_fid_data[i] = cut_fid_data[i % n_bins];
                }
                SyncArray<GHPair> missing_gh(n_partition);
                SyncArray<GHPair> hist(n_bins * n_nodes_in_level);
                party.booster.fbuilder->compute_histogram_in_a_level(l, n_max_splits, n_bins,
                                                                     n_nodes_in_level,
                                                                     hist_fid_data, missing_gh, hist);
                if (fl_param.privacy_tech == "he" && !party.has_label) {
                    party.SendHistogramBatchesEnc(hist, 0); // 0 represents hist
                    party.SendHistogramBatchesEnc(missing_gh, 1); // 1 represents missing_gh
                }
                else {
                    party.SendHistogramBatches(hist, 0); // 0 represents hist
                    party.SendHistogramBatches(missing_gh, 1); // 1 represents missing_gh
                }
                party.SendHistFidBatches(hist_fid);

                if (party.pid == 0)
                    party.TriggerAggregate(n_nodes_in_level);

                vector<BestInfo> bests;
                party.GetBestInfo(bests);

                party.booster.fbuilder->sp.resize(n_nodes_in_level);
                bool updated = false;
                vector<bool> h_s_vector;
                for (auto &best: bests) {
                    if (best.pid != party.pid)
                        continue;
                    updated = true;
                    party.booster.fbuilder->get_split_points_in_a_node(best.nid, best.idx, best.gain, n_nodes_in_level,
                                                                       hist_fid_data, missing_gh, hist);
                    party.booster.fbuilder->update_tree_in_a_node(best.nid);
                    int node_shifted = best.nid + (1 << l) - 1;
                    bool has_split = party.booster.fbuilder->update_ins2node_id_in_a_node(node_shifted);
                    h_s_vector.push_back(has_split);

                    auto nodes_data = party.booster.fbuilder->trees.nodes.host_data();
                    auto sp_data = party.booster.fbuilder->sp.host_data();
                    sp_data[best.nid].split_fea_id = best.global_fid;
                    nodes_data[node_shifted].split_feature_id = best.global_fid;

                    int lch = nodes_data[node_shifted].lch_index;
                    int rch = nodes_data[node_shifted].rch_index;
                    if (fl_param.privacy_tech == "he" && !party.has_label) {
                        fedtree::NodeEncArray nodes;
                        party.OrganizeNodesEnc(nodes, nodes_data[node_shifted]);
                        party.OrganizeNodesEnc(nodes, nodes_data[lch]);
                        party.OrganizeNodesEnc(nodes, nodes_data[rch]);
                        party.SendNodesEnc(nodes);
                    }
                    else {
                        fedtree::NodeArray nodes;
                        party.OrganizeNodes(nodes, nodes_data[node_shifted]);
                        party.OrganizeNodes(nodes, nodes_data[lch]);
                        party.OrganizeNodes(nodes, nodes_data[rch]);
                        party.SendNodes(nodes);
                    }
                    
                    party.SendIns2NodeIDBatches(party.booster.fbuilder->ins2node_id, node_shifted, l);
                }

                party.GetNodes(l);
                party.GetIns2NodeIDBatches();

                bool cont;
                if (updated) {
                    bool has_split = false;
                    for (bool h_s:h_s_vector) {
                        if (h_s) {
                            has_split = true;
                            break;
                        }
                    }
                    cont = party.CheckIfContinue(has_split);
                } else
                    cont = party.CheckIfContinue(false);
                if (!cont) break;
            }
            party.booster.fbuilder->trees.prune_self(param.gamma);
            party.booster.fbuilder->predict_in_training(t);
            if (party.pid == 0)
                party.TriggerPrune(t);
            tree = party.booster.fbuilder->trees;
        }
        party.gbdt.trees.push_back(trees);
        if (party.pid == 0)
            party.TriggerPrintScore();
//        LOG(INFO) << party.booster.metric->get_name() << " = "
//                   << party.booster.metric->get_score(party.booster.fbuilder->get_y_predict());
        LOG(INFO) << "Training round " << round << " end";
    }
}

void distributed_horizontal_train(DistributedParty& party, FLParam &fl_param) {
    // initialization
    if (fl_param.privacy_tech == "he") {
        if (party.pid == 0) {
            party.TriggerHomoInit();
        }
        party.GetPaillier();
    }
    else if(fl_param.privacy_tech == "sa"){
//        if (party.pid == 0){
//            party.TriggerSAInit();
//        }
//        party.GetPG();
        party.dh.generate_public_key();
        party.dh.pid = party.pid;
        party.dh.init_variables(party.n_parties);
        party.SendDHPubKey();
        party.GetDHPubKey();
        party.dh.compute_shared_keys();
    }
    DifferentialPrivacy dp_manager = DifferentialPrivacy();
    if (fl_param.privacy_tech == "dp") {
        LOG(INFO) << "Start DP init";
        dp_manager.init(fl_param);
    }

    int n_bins = fl_param.gbdt_param.max_num_bin;
    if(fl_param.propose_split == "server") {
        // get and send feature range
        vector<vector<float_type>> feature_range(party.get_num_feature());
        for (int i = 0; i < party.get_num_feature(); i++) {
            feature_range[i] = party.get_feature_range_by_feature_index(i);
        }

        party.SendRange(feature_range);
        if (party.pid == 0) {
            //trigger send
            party.TriggerCut(n_bins);
        }
        // GetRange
        party.GetRangeAndSet(n_bins);
        // initialization end
    }
    else if(fl_param.propose_split == "party"){
//        party.booster.fbuilder->cut.get_cut_points_fast(party.dataset, n_bins, party.dataset.n_instances());
        party.SendCutPoints();
        party.GetCutPoints();
    }
    for (int i = 0; i < fl_param.gbdt_param.n_trees; i++) {
        LOG(INFO) << "Training round " << i << " start";
        vector<Tree> trees(fl_param.gbdt_param.tree_per_rounds);
        party.booster.update_gradients();
        if (fl_param.privacy_tech == "dp") {
            // TODO
            // auto gradient_data = party.booster.gradients.host_data();
            // for(int i = 0; i < party.booster.gradients.size(); i++){
            //     dp_manager.clip_gradient_value(gradient_data[i].g);
            // }
        }
        GHPair party_gh = thrust::reduce(thrust::host, party.booster.gradients.host_data(), party.booster.gradients.host_end());
        // send party_gh to server
        party.SendGH(party_gh);
        for (int k = 0; k < fl_param.gbdt_param.tree_per_rounds; k++) {
            party.booster.fbuilder->build_init(party.booster.gradients, k);
            if (party.pid == 0) {
                party.TriggerBuildUsingGH(k);
            }
            for (int d = 0; d < fl_param.gbdt_param.depth; d++) {
                if (fl_param.privacy_tech == "sa"){
                    party.dh.generate_noises();
                    party.SendNoises();
                    party.GetNoises();
                    party.dh.decrypt_noises();
                }
                int n_nodes_in_level = 1 << d;
                int n_max_nodes = 2 << fl_param.gbdt_param.depth;
                int n_column = party.dataset.n_features();
                int n_partition = n_column * n_nodes_in_level;
                int n_bins = party.booster.fbuilder->cut.cut_points_val.size();      
                auto cut_fid_data = party.booster.fbuilder->cut.cut_fid.host_data();
                int n_max_splits = n_max_nodes * n_bins;
                SyncArray<int> hist_fid(n_nodes_in_level * n_bins);
                auto hist_fid_data = hist_fid.host_data();                    
                for (int i = 0; i < hist_fid.size(); i++)
                    hist_fid_data[i] = cut_fid_data[i % n_bins];

                
                SyncArray <GHPair> missing_gh(n_partition);
                SyncArray <GHPair> hist(n_nodes_in_level * n_bins);
                party.booster.fbuilder->compute_histogram_in_a_level(d, n_max_splits, n_bins,
                                                                              n_nodes_in_level,
                                                                              hist_fid_data, missing_gh, hist);
                // TODO encrypt the histogram
                // if (fl_param.privacy_tech == "he") {

                // }
                // append hist
                if (fl_param.privacy_tech == "he") {
                    {
                        TIMED_SCOPE(timerObj, "encrypting time");
                        auto t_start = party.timer.now();
                        party.encrypt_histogram(hist);
                        party.encrypt_histogram(missing_gh);
                        auto t_end = party.timer.now();
                        std::chrono::duration<double> used_time = t_end - t_start;
                        party.enc_time += used_time.count();
                    }
                    party.SendHistogramBatchesEnc(hist, 0);
                    party.SendHistogramBatchesEnc(missing_gh, 1);
                }
                else if(fl_param.privacy_tech == "sa"){
                    party.add_noise_to_histogram(hist);
                    party.add_noise_to_histogram(missing_gh);
                    party.SendHistogramBatches(hist, 0);
                    party.SendHistogramBatches(missing_gh, 1);
                }
                else {
                    party.SendHistogramBatches(hist, 0);
                    party.SendHistogramBatches(missing_gh, 1);
                }
                // send party_hist_fid
                party.SendHistFidBatches(hist_fid);
                if (party.pid == 0) {
                    // send current d
                    party.TriggerCalcTree(d);
                }
                // get split points 
                party.GetSplitPoints();
                if (d == 0) {
                    party.GetRootNode();
                }
                party.booster.fbuilder->update_tree();
                party.booster.fbuilder->update_ins2node_id();
                // check if continue
                if (!party.HCheckIfContinue()) {
                    break;
                }
            }
            // server prune
//            if (party.pid == 0) {
//                party.TriggerPrune(k);
//            }
            Tree &tree = trees[k];
            tree = party.booster.fbuilder->get_tree();
            party.booster.fbuilder->trees.prune_self(fl_param.gbdt_param.gamma);
            party.booster.fbuilder->predict_in_training(k);
            
            
            tree.nodes.resize(party.booster.fbuilder->trees.nodes.size());
            tree.nodes.copy_from(party.booster.fbuilder->trees.nodes);
            // trigger prune
        }
        
        party.gbdt.trees.push_back(trees);
        float score = party.booster.metric->get_score(party.booster.fbuilder->get_y_predict());
        float avg_score = party.GetAvgScore(score);
        LOG(INFO) << "averaged " << party.booster.metric->get_name() << " = "
                  << avg_score;
        LOG(INFO) << "Training round " << i << " end";
    }
}

void distributed_ensemble_train(DistributedParty &party, FLParam &fl_param){
    CHECK_EQ(fl_param.gbdt_param.n_trees % fl_param.n_parties, 0);
    int n_tree_each_party = fl_param.gbdt_param.n_trees / fl_param.n_parties;
    for (int j = 0; j < n_tree_each_party; j++)
        party.booster.boost(party.gbdt.trees);
//    party.SendTrees();
}

#ifdef _WIN32
INITIALIZE_EASYLOGGINGPP
#endif

int main(int argc, char **argv) {
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
//    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);
    
    int pid;
    FLParam fl_param;
    Parser parser;
    if (argc > 2) {
        pid = std::stoi(argv[2]);
        parser.parse_param(fl_param, argc, argv);
    } else {
        printf("Usage: <config file path> <pid>\n");
        exit(0);
    }
    GBDTParam &model_param = fl_param.gbdt_param;
    if(model_param.verbose == 0) {
        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "false");
    }
    else if (model_param.verbose == 1) {
        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
    }
    if (!model_param.profiling) {
        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
    }
    DistributedParty party(grpc::CreateChannel(fl_param.ip_address + ":50051",
                                               grpc::InsecureChannelCredentials()));
    party.n_parties = fl_param.n_parties;
    GBDTParam &param = fl_param.gbdt_param;
    DataSet dataset;
    dataset.load_from_file(param.path, fl_param);
    DataSet test_dataset;
    bool use_global_test_set = !param.test_path.empty();
    if(use_global_test_set)
        test_dataset.load_from_file(param.test_path, fl_param);
    Partition partition;
    vector<DataSet> subsets(fl_param.n_parties);
    std::map<int, vector<int>> batch_idxs;

    if(param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos) {
        int num_class = dataset.label.size();
        if ((param.num_class == 1) && (param.num_class != num_class)) {
            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
            param.num_class = num_class;
        }
        if(param.num_class > 2)
            param.tree_per_rounds = param.num_class;
    }
    else if(param.objective.find("reg:") != std::string::npos){
        param.num_class = 1;
    }
    float train_time = 0;
    if (fl_param.mode == "vertical") {
//        LOG(INFO) << "vertical dir";
        dataset.csr_to_csc();
        if (fl_param.partition) {
            partition.homo_partition(dataset, fl_param.n_parties, false, subsets, batch_idxs);
            party.vertical_init(pid, subsets[pid], fl_param);
        }
        else {
            // calculate batch idxs
            if(use_global_test_set)
                for(int i = 0; i < test_dataset.n_features(); i++)
                    batch_idxs[0].push_back(i);
            party.vertical_init(pid, dataset, fl_param);
        }
        party.BeginBarrier();
        LOG(INFO)<<"training start";
        auto t_start = party.timer.now();
        distributed_vertical_train(party, fl_param);
        auto t_end = party.timer.now();
        std::chrono::duration<float> used_time = t_end - t_start;
        LOG(INFO)<<"training end";
        train_time = used_time.count();
        LOG(INFO) << "train time: " << train_time<<"s";
        if(use_global_test_set)
            party.gbdt.predict_score_vertical(fl_param.gbdt_param, test_dataset, batch_idxs);
    }
    else if (fl_param.mode == "horizontal") {
        // horizontal does not need feature_map parameter
        if (fl_param.partition) {
            partition.homo_partition(dataset, fl_param.n_parties, true, subsets, batch_idxs);
            party.init(pid, subsets[pid], fl_param);
        }
        else {
            party.init(pid, dataset, fl_param);
        }

        party.BeginBarrier();
        LOG(INFO)<<"training start";
        auto t_start = party.timer.now();
        distributed_horizontal_train(party, fl_param);
        auto t_end = party.timer.now();
        std::chrono::duration<float> used_time = t_end - t_start;
        LOG(INFO)<<"training end";
        train_time = used_time.count();
        LOG(INFO) << "train time: " << train_time<<"s";
        if(use_global_test_set)
            party.gbdt.predict_score(fl_param.gbdt_param, test_dataset);
    }
    else if (fl_param.mode == "ensemble"){
        if (fl_param.partition){
            partition.homo_partition(dataset, fl_param.n_parties, fl_param.partition_mode == "horizontal", subsets, batch_idxs, fl_param.seed);
            party.init(pid, subsets[pid], fl_param);
        }
        else{
            party.init(pid, dataset, fl_param);
        }
        party.BeginBarrier();
        LOG(INFO)<<"training start";
        auto t_start = party.timer.now();
        distributed_ensemble_train(party, fl_param);
        auto t_end = party.timer.now();
        std::chrono::duration<float> used_time = t_end - t_start;
        LOG(INFO)<<"training end";
        train_time = used_time.count();
        LOG(INFO) << "train time: " << train_time<<"s";
        if(use_global_test_set)
            party.gbdt.predict_score(fl_param.gbdt_param, test_dataset);

    }
    
    LOG(INFO) << "encryption time:" << party.enc_time << "s";
    parser.save_model(fl_param.gbdt_param.model_path, fl_param.gbdt_param, party.gbdt.trees);
    party.StopServer(train_time);
    return 0;
}
