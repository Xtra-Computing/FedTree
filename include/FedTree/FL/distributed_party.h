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
    void SendNodeEnc(Tree::TreeNode &node_data);

    void OrganizeNodesEnc(fedtree::NodeEncArray &nodes, Tree::TreeNode &node_data);
    void SendNodesEnc(fedtree::NodeEncArray &nodes);
    void OrganizeNodes(fedtree::NodeArray &nodes, Tree::TreeNode &node_data);
    void SendNodes(fedtree::NodeArray &nodes);

    void SendIns2NodeID(SyncArray<int> &ins2node_id, int nid);

    void GetNodes(int l);

    void GetIns2NodeID();

    bool CheckIfContinue(bool cont);

    void TriggerPrune(int t);

    void TriggerPrintScore();

    void SendRange(const vector<vector<float_type>>& ranges);

    void SendCutPoints();

    void GetCutPoints();
    
    void TriggerCut(int n_bins);
    
    void GetRangeAndSet(int n_bins);
    
    void SendGH(GHPair party_gh);
    
    void TriggerBuildUsingGH(int k);

    void TriggerCalcTree(int l);

    void GetRootNode();

    void GetSplitPoints();

    bool HCheckIfContinue();
    
    float GetAvgScore(float score);

    void TriggerHomoInit();

    void TriggerSAInit();

    void GetPaillier();

    void SendHistogramsEnc(const SyncArray<GHPair> &hist, int type);

    void SendHistogramBatches(const SyncArray<GHPair> &hist, int type);

    void SendHistFidBatches(const SyncArray<int> &hist);

    void GetIns2NodeIDBatches();

    void SendIns2NodeIDBatches(SyncArray<int> &ins2node_id, int nid);

    void GetGradientBatches();

    void GetGradientBatchesEnc();

    void SendHistogramBatchesEnc(const SyncArray<GHPair> &hist, int type);
    
    void StopServer(float tot_time);

    void BeginBarrier();

    void SendDHPubKey();
    void GetDHPubKey();
    void SendNoises();
    void GetNoises();
    double comm_time = 0;
    double enc_time = 0;
    double comm_size = 0;
    int n_parties;
    std::chrono::high_resolution_clock timer;

private:
    std::unique_ptr<fedtree::FedTree::Stub> stub_;
};

void distributed_vertical_train(DistributedParty& party, FLParam &fl_param);
void distributed_horizontal_train(DistributedParty& party, FLParam &fl_param);
void distributed_ensemble_train(DistributedParty &party, FLParam &fl_param);
#endif //FEDTREE_DISTRIBUTED_PARTY_H
