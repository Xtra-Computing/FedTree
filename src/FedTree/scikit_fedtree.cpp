//
// Created by Kelly Yung on 2021/9/23.
//

#include <FedTree/FL/FLtrainer.h>
#include <FedTree/FL/FLparam.h>
#include <FedTree/Tree/GBDTParam.h>


extern "C" {

    void set_logger(int verbose) {
        if(verbose == 0) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "false");
        }
        else if (verbose == 1) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        }
    }

    void fit (int n_parties, int partition, float alpha, int n_hori, int n_verti, char *mode, char *partition_mode, char *privacy_tech, char *propose_split, char *merge_histogram, float variance, float privacy_budget,
              int depth, int n_trees, float min_child_weight, float lambda, float gamma, float column_sampling_rate, int verbose, int bagging, int n_parallel_trees, float learning_rate,
              char *objective, int num_class, int n_device, int max_num_bin, char *metric,
              // DataSet info
              int row_size, float *val, int *row_ptr, int *col_ptr, float *label) {

        // Initialize model params
        FLParam fl_param;
        fl_param.n_parties = n_parties;
        fl_param.partition = partition;
        fl_param.alpha = alpha;
        fl_param.n_hori = n_hori;
        fl_param.n_verti = n_verti;
        fl_param.mode = mode;
        fl_param.partition_mode = mode;
        fl_param.privacy_tech = privacy_tech;
        fl_param.propose_split = propose_split;
        fl_param.merge_histogram = merge_histogram;
        fl_param.variance = variance;
        fl_param.privacy_budget = privacy_budget;

        GBDTParam gbdt_param;
        gbdt_param.depth = depth;
        gbdt_param.n_trees = n_trees;
        gbdt_param.min_child_weight = min_child_weight;
        gbdt_param.lambda = lambda;
        gbdt_param.gamma = gamma;
        gbdt_param.column_sampling_rate = column_sampling_rate;
        gbdt_param.verbose = verbose;
        gbdt_param.bagging = bagging;
        gbdt_param.n_parallel_trees = n_parallel_trees;
        gbdt_param.learning_rate = learning_rate;
        gbdt_param.objective = objective;
        gbdt_param.num_class = num_class;
        gbdt_param.tree_method = "hist";
        gbdt_param.n_device = n_device;
        gbdt_param.tree_per_rounds = 1;
        gbdt_param.max_num_bin = max_num_bin;
        gbdt_param.metric = metric;
        gbdt_param.rt_eps = 1e-6;
        fl_param.gbdt_param = gbdt_param;

        set_logger(verbose);
        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");

        DataSet training_set;
        training_set.load_from_sparse(row_size, val, row_ptr, col_ptr, label, group, num_group, model_param);
        *num_class = gbdt_param.num_class;

        // Partition the dataset
        int n_parties = fl_param.n_parties;
        vector<DataSet> subsets(n_parties);
        vector<SyncArray<bool>> feature_map(n_parties);
        std::map<int, vector<int>> batch_idxs;
        Partition partition;
        if (fl_param.partition == true) {
            if (fl_param.partition_mode == "hybrid") {
                if (fl_param.mode == "horizontal")
                    CHECK_EQ(fl_param.n_verti, 1);
                if (fl_param.mode == "vertical")
                    CHECK_EQ(fl_param.n_hori, 1);
                partition.horizontal_vertical_dir_partition(training_set, n_parties, fl_param.alpha, feature_map, subsets,
                                                            fl_param.n_hori, fl_param.n_verti);
            }
        }else if (fl_param.partition_mode == "vertical") {
            dataset.csr_to_csc();
            partition.homo_partition(dataset, n_parties, false, subsets, batch_idxs);
        }else if (fl_param.partition_mode == "horizontal") {
            partition.homo_partition(dataset, n_parties, true, subsets, batch_idxs);
        }
        // Update tree per rounds to match with number of class
        int num_class = dataset.label.size();
        if (param.num_class != num_class) {
            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
            fl_param.gbdt_param.num_class = num_class;
        }
        fl_param.gbdt_param.tree_per_rounds = param.num_class;

        // Initialize parties
        vector<Party> parties(n_parties);
        vector<int> n_instances_per_party(n_parties);
        LOG(INFO)<<"initialize parties";
        for(int i = 0; i < n_parties; i++){
            parties[i].init(i, subsets[i], fl_param, feature_map[i]);
            n_instances_per_party[i] = subsets[i].n_instances();
        }

        // Initialize server
        LOG(INFO) << "initialize server";
        Server server;
        if (fl_param.mode == "vertical") {
            server.vertical_init(fl_param, dataset.n_instances(), n_instances_per_party, dataset.y, dataset.label);
        }else if (fl_param.mode == "horizontal") {
            server.horizontal_init(fl_param, dataset.n_instances(), n_instances_per_party, dataset);
        }else {
            server.init(fl_param, dataset.n_instances(), n_instances_per_party);
        }

        // Run different training methods based on mode
        FLtrainer trainer;
        if(fl_param.mode == "hybrid") {
            trainer.hybrid_fl_trainer(parties, server, fl_param);
        } else if(fl_param.mode == "ensemble") {
            trainer.ensemble_trainer(parties, server, fl_param);
        }else if(fl_param.mode == "solo") {
            trainer.solo_trainer(parties, fl_param);
        }else if(fl_param.mode == "centralized") {
            GBDT gbdt;
            gbdt.train(fl_param.gbdt_param, training_set);
        }else if (fl_param.mode == "vertical") {
            trainer.vertical_fl_trainer(parties, server, fl_param);
        }else if (fl_param.mode == "horizontal") {
            trainer.horizontal_fl_trainer(parties, server, fl_param);
        }

        // Return boosted model

    }


}