//
// Created by 韩雨萱 on 11/4/21.
//

#include "FedTree/FL/distributed_party.h"
#include "FedTree/FL/partition.h"
#include "FedTree/parser.h"

void DistributedParty::TriggerUpdateGradients() {
    fedtree::PID id;
    fedtree::Ready ready;
    grpc::ClientContext context;
    grpc::Status status = stub_->TriggerUpdateGradients(&context, id, &ready);
    if (status.ok()) {
        LOG(DEBUG) << "Triggered the server to update gradients.";
    } else {
        LOG(DEBUG) << "TriggerUpdateGradients rpc failed.";
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
        LOG(DEBUG) << "TriggerBuildInit rpc failed.";
    }
}

void DistributedParty::GetGradients() {
    fedtree::PID id;
    fedtree::GHPair gh;
    grpc::ClientContext context;
    id.set_id(pid);
    LOG(DEBUG) << "Receiving gradients from the server.";

    std::unique_ptr<grpc::ClientReader<fedtree::GHPair> > reader(stub_->GetGradients(&context, id));

    auto booster_gradients_data = booster.gradients.host_data();
    int i = 0;
    while (reader->Read(&gh)) {
        booster_gradients_data[i] = {static_cast<float_type>(gh.g()), static_cast<float_type>(gh.h())};
        i++;
    }

    grpc::Status status = reader->Finish();
    if (status.ok()) {
        LOG(DEBUG) << "All gradients received.";
    } else {
        LOG(DEBUG) << "GetGradients rpc failed.";
    }
}

void DistributedParty::SendDatasetInfo(int n_bins, int n_columns) {
    fedtree::DatasetInfo datasetInfo;
    fedtree::PID id;
    datasetInfo.set_n_bins(n_bins);
    datasetInfo.set_n_columns(n_columns);
    grpc::ClientContext context;
    context.AddMetadata("pid", std::to_string(pid));
    grpc::Status status = stub_->SendDatasetInfo(&context, datasetInfo, &id);
    if (status.ok()) {
        LOG(DEBUG) << "Dataset info sent.";
    } else {
        LOG(DEBUG) << "SendDatasetInfo rpc failed.";
    }
}

void DistributedParty::SendHistograms(const SyncArray<GHPair> &hist, int type) {
    fedtree::PID id;
    grpc::ClientContext context;
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
    if (status.ok()) {
        LOG(DEBUG) << "All " << type << " sent.";
    } else {
        LOG(DEBUG) << "SendHistograms rpc failed.";
    }
}

void DistributedParty::SendHistFid(const SyncArray<int> &hist_fid) {
    fedtree::PID id;
    grpc::ClientContext context;
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
    if (status.ok()) {
        LOG(DEBUG) << "All hist_fid sent.";
    } else {
        LOG(DEBUG) << "SendHistograms rpc failed.";
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
        LOG(DEBUG) << "TriggerAggregate rpc failed.";
        return false;
    } else if (!ready.ready()) {
        LOG(DEBUG) << "Server has not received all histograms.";
        return false;
    } else {
        return true;
    }
}

void DistributedParty::GetBestInfo(vector<BestInfo> &bests) {
    fedtree::PID id;
    fedtree::BestInfo best;
    grpc::ClientContext context;
    id.set_id(pid);
    std::cout << "Receiving best info from the server." << std::endl;

    std::unique_ptr<grpc::ClientReader<fedtree::BestInfo> > reader(stub_->GetBestInfo(&context, id));
    while (reader->Read(&best)) {
        bests.push_back({best.pid(), best.nid(), best.idx(), best.global_fid(), static_cast<float>(best.gain())});
    }
    grpc::Status status = reader->Finish();
    if (status.ok()) {
        std::cout << "All nodes updated using best info." << std::endl;
    } else {
        std::cout << "GetBestInfo rpc failed." << std::endl;
    }
}

void DistributedParty::SendNode(Tree::TreeNode &node_data) {
    fedtree::Node node;
    fedtree::PID id;
    grpc::ClientContext context;
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
    if (status.ok()) {
        std::cout << "Node sent." << std::endl;
    } else {
        std::cout << "SendNodes rpc failed." << std::endl;
    }
}

void DistributedParty::SendIns2NodeID(SyncArray<int> &ins2node_id, int nid) {
    fedtree::PID id;
    grpc::ClientContext context;
    context.AddMetadata("pid", std::to_string(pid));
    std::unique_ptr<grpc::ClientWriter<fedtree::Ins2NodeID> > writer(
            stub_->SendIns2NodeID(&context, &id));

    auto ins2node_id_data = ins2node_id.host_data();
    for (int i = 0; i < ins2node_id.size(); ++i) {
        if (ins2node_id_data[i] >= 2 * nid + 1 && ins2node_id_data[i] <= 2 * nid + 2) {
            fedtree::Ins2NodeID i2n;
            i2n.set_iid(i);
            i2n.set_nid(ins2node_id_data[i]);
            if (!writer->Write(i2n))
                break;
        }
    }
    writer->WritesDone();
    grpc::Status status = writer->Finish();
    if (status.ok()) {
        LOG(DEBUG) << "ins2node_id of the current node sent.";
    } else {
        LOG(DEBUG) << "SendIns2NodeID rpc failed.";
    }
}

void DistributedParty::GetNodes(int l) {
    fedtree::PID id;
    fedtree::Node node;
    grpc::ClientContext context;
    context.AddMetadata("l", std::to_string(l));
    id.set_id(pid);
    std::cout << "Receiving nodes from the server." << std::endl;

    std::unique_ptr<grpc::ClientReader<fedtree::Node> > reader(stub_->GetNodes(&context, id));
    while (reader->Read(&node)) {
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
        nodes_data[nid].sum_gh_pair.g = node.sum_gh_pair_g();
        nodes_data[nid].sum_gh_pair.h = node.sum_gh_pair_h();
        nodes_data[nid].n_instances = node.n_instances();
    }
    grpc::Status status = reader->Finish();
    if (status.ok()) {
        std::cout << "All nodes received." << std::endl;
    } else {
        std::cout << "GetNodes rpc failed." << std::endl;
    }
}

void DistributedParty::GetIns2NodeID() {
    fedtree::PID id;
    fedtree::Ins2NodeID i2n;
    grpc::ClientContext context;
    id.set_id(pid);
    std::cout << "Receiving ins2node_id from the server." << std::endl;

    auto ins2node_id_data = booster.fbuilder->ins2node_id.host_data();
    std::unique_ptr<grpc::ClientReader<fedtree::Ins2NodeID> > reader(stub_->GetIns2NodeID(&context, id));
    while (reader->Read(&i2n)) {
        int iid = i2n.iid();
        int nid = i2n.nid();
        ins2node_id_data[iid] = nid;
    }
    grpc::Status status = reader->Finish();
    if (status.ok()) {
        std::cout << "All ins2node_id received." << std::endl;
    } else {
        std::cout << "GetIns2NodeID rpc failed." << std::endl;
    }
}

bool DistributedParty::CheckIfContinue(bool cont) {
    fedtree::PID id;
    fedtree::Ready ready;
    grpc::ClientContext context;
    context.AddMetadata("cont", std::to_string(cont));
    id.set_id(pid);
    grpc::Status status = stub_->CheckIfContinue(&context, id, &ready);
    if (!status.ok()) {
        LOG(DEBUG) << "CheckIfContinue rpc failed.";
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
        LOG(DEBUG) << "TriggerPrune rpc failed.";
    }
}

void distributed_vertical_train(DistributedParty& party, FLParam &fl_param) {
    GBDTParam &param = fl_param.gbdt_param;
    party.SendDatasetInfo(party.booster.fbuilder->cut.cut_points_val.size(), party.dataset.n_features());
    for (int round = 0; round < param.n_trees; round++) {
        if (party.pid == 0)
            party.TriggerUpdateGradients();
        party.GetGradients();
        for (int t = 0; t < param.tree_per_rounds; t++) {
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
                party.SendHistograms(hist, 0); // 0 represents hist
                party.SendHistograms(missing_gh, 1); // 1 represents missing_gh
                party.SendHistFid(hist_fid);

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
                    party.SendNode(nodes_data[node_shifted]);
                    party.SendNode(nodes_data[lch]);
                    party.SendNode(nodes_data[rch]);
                    party.SendIns2NodeID(party.booster.fbuilder->ins2node_id, node_shifted);
                }

                party.GetNodes(l);
                party.GetIns2NodeID();

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
        }
        LOG(DEBUG) << party.booster.metric->get_name() << " = "
                   << party.booster.metric->get_score(party.booster.fbuilder->get_y_predict());
    }
}

void distributed_horizontal_train(DistributedParty& party, FLParam &fl_param) {

}

int main(int argc, char **argv) {
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

    DistributedParty party(grpc::CreateChannel("localhost:50051",
                                               grpc::InsecureChannelCredentials()));

    GBDTParam &model_param = fl_param.gbdt_param;
    DataSet dataset;
    dataset.load_from_file(model_param.path, fl_param);
    Partition partition;
    vector<DataSet> subsets(fl_param.n_parties);
    std::map<int, vector<int>> batch_idxs;
    if (fl_param.mode == "vertical") {
        LOG(INFO) << "vertical dir";
        dataset.csr_to_csc();
        partition.homo_partition(dataset, fl_param.n_parties, false, subsets, batch_idxs);
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

        party.vertical_init(pid, subsets[pid], fl_param);
        distributed_vertical_train(party, fl_param);
    }
    else if (fl_param.mode == "horizontal") {
        // draft
        LOG(INFO) << "horizontal dir, developing, stop";
        // subset division
        // party.init(pid, subsets[pid], fl_param, );
        // distributed_horizontal_train(party, fl_param);
    }
    

    return 0;
}