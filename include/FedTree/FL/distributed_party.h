//
// Created by 韩雨萱 on 9/4/21.
//

#ifndef FEDTREE_DISTRIBUTED_PARTY_H
#define FEDTREE_DISTRIBUTED_PARTY_H

#include <grpcpp/grpcpp.h>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include "../../../src/FedTree/grpc/fedtree.grpc.pb.h"
#include "party.h"


class DistributedParty : public Party {
public:
    DistributedParty(std::shared_ptr<grpc::Channel> channel)
            : stub_(fedtree::FedTree::NewStub(channel)) {};

    void TriggerUpdateGradients();

    void TriggerBuildInit(int t);

    void GetGradients();

    void SendDatasetInfo(int n_bins, int n_columns);

    void SendHistograms(const SyncArray<GHPair> &hist, int type);

    void SendHistFid(const SyncArray<int> &hist_fid);

    bool TriggerAggregate(int n_nodes_in_level);

    void GetBestInfo(vector<BestInfo> &bests);

    void SendNode(Tree::TreeNode &node_data);

    void SendIns2NodeID(SyncArray<int> &ins2node_id, int nid);

    void GetNodes(int l);

    void GetIns2NodeID();

    bool CheckIfContinue(bool cont);

    void TriggerPrune(int t);

    // void SendRange(const vector<vector<float>>& ranges);
    // void TriggerCut(int n_bins);
    // void GetRangeAndSet(int n_bins);

private:
    std::unique_ptr<fedtree::FedTree::Stub> stub_;
};

#endif //FEDTREE_DISTRIBUTED_PARTY_H
