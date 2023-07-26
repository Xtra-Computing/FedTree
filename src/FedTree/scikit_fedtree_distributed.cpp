#include "FedTree/FL/FLparam.h"
#include "FedTree/FL/distributed_party.h"
#include "FedTree/FL/distributed_server.h"
#include "FedTree/FL/partition.h"
#include "FedTree/Tree/tree.h"
#include "FedTree/common.h"
#include "FedTree/util/log.h"
#include <string>
#include <thread>

extern "C" {
extern void set_logger(int verbose);

void fit_distributed(int pid, char *ip_address, int port, int n_parties,
                     int partition, float alpha, int n_hori, int n_verti,
                     char *mode, char *partition_mode, char *privacy_tech,
                     char *propose_split, char *merge_histogram, float variance,
                     float privacy_budget, int depth, int n_trees,
                     float min_child_weight, float lambda, float gamma,
                     float column_sampling_rate, int verbose, int bagging,
                     int n_parallel_trees, float learning_rate, char *objective,
                     int *num_class, int n_device, int max_num_bin, int seed,
                     float ins_bagging_fraction, int reorder_label,
                     float constant_h,
                     // DataSet info
                     int row_size, float_type *val, int *row_ptr, int *col_ptr,
                     float_type *label,
                     // Tree info
                     char *tree_method, Tree *&model, int *tree_per_iter,
                     float_type *group_label, int *group, int num_group) {
    bool is_server = pid == -1;
    // Initialize model params
    LOG(INFO) << "Initialize FL parameters";
    FLParam fl_param;
    fl_param.n_parties = n_parties;
    fl_param.partition = partition == 1 ? true : false;
    fl_param.alpha = alpha;
    fl_param.n_hori = n_hori;
    fl_param.n_verti = n_verti;
    fl_param.mode = mode;
    fl_param.partition_mode = partition_mode;
    fl_param.privacy_tech = privacy_tech;
    fl_param.propose_split = propose_split;
    fl_param.merge_histogram = merge_histogram;
    fl_param.variance = variance;
    fl_param.privacy_budget = privacy_budget;
    fl_param.seed = seed;
    fl_param.ins_bagging_fraction = ins_bagging_fraction;
    fl_param.ip_address = ip_address;

    GBDTParam gbdt_param;
    gbdt_param.depth = depth;
    gbdt_param.n_trees = n_trees;
    gbdt_param.min_child_weight = min_child_weight;
    gbdt_param.lambda = lambda;
    gbdt_param.gamma = gamma;
    gbdt_param.column_sampling_rate = column_sampling_rate;
    gbdt_param.verbose = verbose;
    gbdt_param.bagging = bagging == 1 ? true : false;
    gbdt_param.n_parallel_trees = n_parallel_trees;
    gbdt_param.learning_rate = learning_rate;
    gbdt_param.objective = objective;
    gbdt_param.num_class = num_class[0];
    gbdt_param.tree_method = "hist";
    gbdt_param.n_device = n_device;
    gbdt_param.tree_per_round = 1;
    gbdt_param.max_num_bin = max_num_bin;
    gbdt_param.rt_eps = 1e-6;
    gbdt_param.metric = "default";
    gbdt_param.reorder_label = reorder_label;
    gbdt_param.constant_h = constant_h;

    if (fl_param.privacy_tech == "dp" && gbdt_param.constant_h == 0)
        gbdt_param.constant_h = 1.0;

    if (fl_param.n_hori == -1) {
        if (fl_param.mode == "horizontal") {
            fl_param.n_hori = fl_param.n_parties;
        } else
            fl_param.n_hori = 1;
    }
    if (fl_param.n_verti == -1) {
        if (fl_param.mode == "vertical") {
            fl_param.n_verti = fl_param.n_parties;
        } else
            fl_param.n_verti = 1;
    }

    set_logger(verbose);
    el::Loggers::reconfigureAllLoggers(
        el::ConfigurationType::PerformanceTracking, "false");

    num_class[0] = gbdt_param.num_class;
    fl_param.gbdt_param = gbdt_param;

    GBDTParam &param = fl_param.gbdt_param;

    std::string server_addr =
        std::string(ip_address) + ":" + std::to_string(port);

    if (is_server) {
        DistributedServer server;

        if (fl_param.mode == "vertical") {
            LOG(INFO) << "Load Sparse Data to Training Set";
            DataSet dataset;
            dataset.load_from_sparse(row_size, val, row_ptr, col_ptr, label,
                                     group, num_group, gbdt_param);
            server.VerticalInitVectors(n_parties);
            server.distributed_vertical_init(fl_param, dataset.n_instances(),
                                             dataset.y, dataset.label);
        } else if (fl_param.mode == "horizontal") {
            server.HorizontalInitVectors(n_parties);
            server.param = fl_param;
            server.horizontal_init(fl_param);
            server.booster.fbuilder->party_containers_init(fl_param.n_parties);
        }

        LOG(INFO) << "server init completed.";

        grpc::ServerBuilder builder;
        builder.AddListeningPort(server_addr,
                                 grpc::InsecureServerCredentials());
        builder.RegisterService(&server);
        std::unique_ptr<grpc::Server> server_ptr(builder.BuildAndStart());
        LOG(INFO) << "Server listening on " << server_addr;

        // Start a thread to shutdown the server.
        auto thread = std::thread([&server, &server_ptr]() {
            server.block_until_shutdown();
            LOG(INFO) << "Server shutdown";
            server_ptr->Shutdown();
        });
        thread.join();

        server_ptr->Wait();
    } else {
        DistributedParty party(grpc::CreateChannel(
            server_addr, grpc::InsecureChannelCredentials()));
        party.n_parties = n_parties;

        LOG(INFO) << "Load Sparse Data to Training Set";
        DataSet dataset;
        dataset.load_from_sparse(row_size, val, row_ptr, col_ptr, label, group,
                                 num_group, gbdt_param);

        vector<DataSet> subsets(fl_param.n_parties);
        std::map<int, vector<int>> batch_idxs;
        Partition partition;
        float train_time;

        GBDTParam &model_param = fl_param.gbdt_param;

        if (model_param.objective.find("multi:") != std::string::npos ||
            model_param.objective.find("binary:") != std::string::npos) {
            int num_class = dataset.label.size();
            if ((model_param.num_class == 1) &&
                (model_param.num_class != num_class)) {
                LOG(INFO) << "updating number of classes from "
                          << model_param.num_class << " to " << num_class;
                model_param.num_class = num_class;
            }
            if (model_param.num_class > 2)
                model_param.tree_per_round = model_param.num_class;
        } else if (model_param.objective.find("reg:") != std::string::npos) {
            model_param.num_class = 1;
        }

        if (fl_param.mode == "vertical") {
            dataset.csr_to_csc();
            if (fl_param.partition) {
                partition.homo_partition(dataset, fl_param.n_parties, false,
                                         subsets, batch_idxs);
                party.vertical_init(pid, subsets[pid], fl_param);
            } else {
                party.vertical_init(pid, dataset, fl_param);
            }

            party.BeginBarrier();

            LOG(INFO) << "training started.";
            auto t_start = party.timer.now();
            distributed_vertical_train(party, fl_param);
            auto t_end = party.timer.now();

            std::chrono::duration<float> used_time = t_end - t_start;
            LOG(INFO) << "training end";
            float train_time = used_time.count();
            LOG(INFO) << "train time: " << train_time << "s";
        } else if (fl_param.mode == "horizontal") {
            if (fl_param.partition) {
                partition.homo_partition(dataset, fl_param.n_parties, true,
                                         subsets, batch_idxs);
                party.init(pid, subsets[pid], fl_param);
            } else {
                party.init(pid, dataset, fl_param);
            }

            party.BeginBarrier();

            LOG(INFO) << "training start";
            auto t_start = party.timer.now();
            distributed_horizontal_train(party, fl_param);
            auto t_end = party.timer.now();

            std::chrono::duration<float> used_time = t_end - t_start;
            LOG(INFO) << "training end";
            train_time = used_time.count();
            LOG(INFO) << "train time: " << train_time << "s";
        } else if (fl_param.mode == "ensemble") {
            if (fl_param.partition) {
                partition.homo_partition(dataset, fl_param.n_parties,
                                         fl_param.partition_mode ==
                                             "horizontal",
                                         subsets, batch_idxs, fl_param.seed);
                party.init(pid, subsets[pid], fl_param);
            } else {
                party.init(pid, dataset, fl_param);
            }
            party.BeginBarrier();

            LOG(INFO) << "training start";
            auto t_start = party.timer.now();
            distributed_ensemble_train(party, fl_param);
            auto t_end = party.timer.now();

            std::chrono::duration<float> used_time = t_end - t_start;
            LOG(INFO) << "training end";
            train_time = used_time.count();
            LOG(INFO) << "train time: " << train_time << "s";
        }
        LOG(INFO) << "encryption time:" << party.enc_time << "s";
        party.StopServer(train_time);
        auto &boosted_model = party.gbdt.trees;

        *tree_per_iter = (int)(boosted_model[0].size());
        model = new Tree[n_trees * (*tree_per_iter)];
        for (int i = 0; i < n_trees; i++) {
            for (int j = 0; j < *tree_per_iter; j++) {
                model[i * (*tree_per_iter) + j] = boosted_model[i][j];
            }
        }
        for (int i = 0; i < dataset.label.size(); ++i) {
            group_label[i] = dataset.label[i];
        }
    }
}
}