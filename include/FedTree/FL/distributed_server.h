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

    grpc::Status SendNodes(grpc::ServerContext *context, const fedtree::NodeArray *nodes, fedtree::PID *id) override;

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

    grpc::Status TriggerPrintScore(grpc::ServerContext *context, const fedtree::PID *pid, fedtree::Ready *ready) override;

    grpc::Status SendRange(grpc::ServerContext* context, grpc::ServerReader<fedtree::GHPair>* reader,
                                fedtree::PID* response) override;
    
    grpc::Status TriggerCut(grpc::ServerContext* context, const fedtree::PID* request,
                                fedtree::Ready* response) override;
    
    grpc::Status GetRange(grpc::ServerContext* context, const fedtree::PID* request,
                                grpc::ServerWriter<fedtree::GHPair>* writer) override;
    

    grpc::Status SendGH(grpc::ServerContext* context, const fedtree::GHPair* request, fedtree::PID* response) override;

    grpc::Status SendDHPubKey(grpc::ServerContext* context, const fedtree::DHPublicKey* request, fedtree::PID* response) override;

    grpc::Status GetDHPubKeys(grpc::ServerContext* context, const fedtree::PID* request, grpc::ServerWriter<fedtree::DHPublicKeys>* response) override;

    grpc::Status SendNoises(grpc::ServerContext* context, const fedtree::SANoises* request, fedtree::PID* response) override;

    grpc::Status GetNoises(grpc::ServerContext* context, const fedtree::PID* request, grpc::ServerWriter<fedtree::SANoises>* response) override;

    grpc::Status SendCutPoints(grpc::ServerContext* context, const fedtree::CutPoints* request, fedtree::PID* response) override;

    grpc::Status GetCutPoints(grpc::ServerContext* context, const fedtree::PID* request, grpc::ServerWriter<fedtree::CutPoints>* response) override;

    grpc::Status TriggerBuildUsingGH(grpc::ServerContext* context, const fedtree::PID* request, fedtree::Ready* response) override;
    
    grpc::Status ScoreReduce(grpc::ServerContext* context, const fedtree::Score* request, fedtree::Score* response) override;
    
    grpc::Status TriggerCalcTree(grpc::ServerContext* context, const fedtree::PID* request, fedtree::Ready* response) override;

    grpc::Status GetSplitPoints(grpc::ServerContext* context, const fedtree::PID* request,
                                grpc::ServerWriter<fedtree::SplitPoint>* writer) override;

    grpc::Status GetRootNode(grpc::ServerContext* context, const fedtree::PID *request, fedtree::Node* response) override;

    grpc::Status HCheckIfContinue(grpc::ServerContext *context, const fedtree::PID *pid,
                                 fedtree::Ready *ready) override;

    grpc::Status TriggerHomoInit(grpc::ServerContext *context, const fedtree::PID *request, fedtree::Ready *response) override;

    grpc::Status TriggerSAInit(grpc::ServerContext *context, const fedtree::PID *request, fedtree::Ready *response) override;

    grpc::Status GetPaillier(grpc::ServerContext *context, const fedtree::PID *request, fedtree::Paillier * response) override;
    
    grpc::Status SendHistogramsEnc(grpc::ServerContext *context, grpc::ServerReader<fedtree::GHPairEnc> *reader,
                                fedtree::PID *id) override;

    grpc::Status SendHistogramBatches(grpc::ServerContext *context, grpc::ServerReader<fedtree::GHBatch> *reader,
                                fedtree::PID *id) override;
    
    grpc::Status SendHistFidBatches(grpc::ServerContext *context, grpc::ServerReader<fedtree::FIDBatch> *reader,
                                fedtree::PID *id) override;
    
    grpc::Status GetIns2NodeIDBatches(grpc::ServerContext *context, const fedtree::PID *id,
                                grpc::ServerWriter<fedtree::Ins2NodeIDBatch> *writer) override;
    
    grpc::Status SendIns2NodeIDBatches(grpc::ServerContext *context, grpc::ServerReader<fedtree::Ins2NodeIDBatch> *reader,
                                fedtree::PID *id) override;

    grpc::Status GetGradientBatches(grpc::ServerContext *context, const fedtree::PID *id,
                              grpc::ServerWriter<fedtree::GHBatch> *writer) override;
    
    grpc::Status SendHistogramBatchesEnc(grpc::ServerContext *context, grpc::ServerReader<fedtree::GHEncBatch> *reader,
                                fedtree::PID *id) override;
    
    grpc::Status StopServer(grpc::ServerContext *context, const fedtree::PID *request,
                                fedtree::Score *ready) override;
    
    grpc::Status BeginBarrier(grpc::ServerContext *context, const fedtree::PID *request, 
                                fedtree::Ready *ready) override;
    
    grpc::Status GetGradientBatchesEnc(grpc::ServerContext *context, const fedtree::PID *request,
                                grpc::ServerWriter<fedtree::GHEncBatch> *writer) override;
    
    grpc::Status SendNodeEnc(grpc::ServerContext *context, const fedtree::NodeEnc *node,
                          fedtree::PID *id) override;

    grpc::Status SendNodesEnc(grpc::ServerContext *context, const fedtree::NodeEncArray *nodes, fedtree::PID *id) override;

    void VerticalInitVectors(int n_parties);

    void HorizontalInitVectors(int n_parties);

    vector<int> n_bins_per_party;
    vector<int> n_columns_per_party;
    void distributed_vertical_init(FLParam &param, int n_total_instances, vector<float_type> y, vector<float_type> label) {
        this->local_trees.resize(param.n_parties);
        this->param = param;
        this->model_param = param.gbdt_param;
        this->n_total_instances = n_total_instances;
        this->n_bins_per_party.resize(param.n_parties);
        this->n_columns_per_party.resize(param.n_parties);
        this->global_trees.trees.clear();
        this->has_label.resize(param.n_parties);
        dataset.y = y;
        dataset.n_features_ = 0;
        dataset.label = label;
        booster.init(dataset, param.gbdt_param);
        booster.fbuilder->party_containers_init(param.n_parties);
    }
    // for profiling
    vector<double> party_wait_times;

private:
    // mutex
    std::mutex mutex;
    // for stop
    vector<bool> stoppable;
    vector<float> party_tot_times;
    vector<float> party_comm_times;
    vector<float> party_enc_times;
    vector<fedtree::GHEncBatch> tmp_gradients;
    
    double dec_time = 0;
    double enc_time = 0;
    // end
    vector<int> cont_votes;
    vector<BestInfo> best_infos;
    vector<int> hists_received;
    vector<int> missing_gh_received;
    vector<int> hist_fid_received;
    int cur_round = 1;
    int n_nodes_received = 0;
    bool update_gradients_success = false;
    bool aggregate_success = false;

    // for horizontal
    vector<vector<GHPair>> party_feature_range;
    vector<int> range_received;
    bool range_success = false;

    vector<int> party_gh_received;
    vector<GHPair> party_ghs;
    int gh_rounds = 1;
    bool build_gh_success = false;
    vector<int> party_DHKey_received;
    vector<int> party_noises_received;
    vector<int> parties_cut_points_received;
    int noise_rounds = 1;
    int noise_cnt = 0;

    bool calc_success = false;
    
    vector<int> score_received;
    vector<float> party_scores;
    bool score_success = false;
    int score_rounds = 1;

    bool homo_init_success = false;

    int info_cnt = 0;
    int cnt = 0;
    int gh_cnt = 0;
    int sp_cnt = 0;
    int vote_cnt = 0;


};

#endif //FEDTREE_DISTRIBUTED_SERVER_H
