//
// Created by 韩雨萱 on 11/4/21.
//

#ifndef FEDTREE_DISTRIBUTED_SERVER_H
#define FEDTREE_DISTRIBUTED_SERVER_H

#include <grpcpp/grpcpp.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include "FedTree/FL/server.h"
#include "FedTree/FL/comm_helper.h"

#include <iostream>
#include <memory>
#include <string>

#ifdef BAZEL_BUILD
#include "examples/protos/fedtree.grpc.pb.h"
#else

#include "../../../src/FedTree/grpc/fedtree.grpc.pb.h"

#endif


class DistributedServer final : public Server, public fedtree::FedTree::Service {
public:
    grpc::Status TriggerUpdateGradients(grpc::ServerContext *context, const fedtree::PID *pid,
                                        fedtree::Ready *ready) override;

    grpc::Status TriggerBuildInit(grpc::ServerContext *context, const fedtree::PID *pid,
                                  fedtree::Ready *ready) override;

    grpc::Status GetGradients(grpc::ServerContext *context, const fedtree::PID *id,
                              grpc::ServerWriter<fedtree::GHPair> *writer) override;

    grpc::Status SendDatasetInfo(grpc::ServerContext *context, const fedtree::DatasetInfo *datasetInfo,
                                 fedtree::PID *pid) override;

    grpc::Status SendHistograms(grpc::ServerContext *context, grpc::ServerReader<fedtree::GHPair> *reader,
                                fedtree::PID *id) override;

    grpc::Status SendHistFid(grpc::ServerContext *context, grpc::ServerReader<fedtree::FID> *reader,
                             fedtree::PID *id) override;

    grpc::Status TriggerAggregate(grpc::ServerContext *context, const fedtree::PID *pid,
                                  fedtree::Ready *ready) override;

    grpc::Status GetBestInfo(grpc::ServerContext *context, const fedtree::PID *id,
                             grpc::ServerWriter<fedtree::BestInfo> *writer) override;

    grpc::Status SendNode(grpc::ServerContext *context, const fedtree::Node *node,
                          fedtree::PID *id) override;

    grpc::Status SendIns2NodeID(grpc::ServerContext *context, grpc::ServerReader<fedtree::Ins2NodeID> *reader,
                                fedtree::PID *id) override;

    grpc::Status GetNodes(grpc::ServerContext *context, const fedtree::PID *id,
                          grpc::ServerWriter<fedtree::Node> *writer) override;

    grpc::Status GetIns2NodeID(grpc::ServerContext *context, const fedtree::PID *id,
                               grpc::ServerWriter<fedtree::Ins2NodeID> *writer) override;

    grpc::Status CheckIfContinue(grpc::ServerContext *context, const fedtree::PID *pid,
                                 fedtree::Ready *ready) override;

    grpc::Status TriggerPrune(grpc::ServerContext *context, const fedtree::PID *pid,
                              fedtree::Ready *ready) override;

    void InitVectors(int n_parties);

private:
    vector<int> cont_votes;
    vector<BestInfo> best_infos;
    vector<int> hists_received;
    vector<int> missing_gh_received;
    vector<int> hist_fid_received;
    int cur_round = 1;
    int n_nodes_received = 0;
    bool update_gradients_success = false;
    bool aggregate_success = false;
};

#endif //FEDTREE_DISTRIBUTED_SERVER_H
