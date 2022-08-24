//
// Created by yuxuan on 11/4/21.
//

#include "FedTree/FL/distributed_party.h"
#include "FedTree/FL/partition.h"
#include "FedTree/parser.h"
#include "FedTree/DP/differential_privacy.h"
#include <sstream>


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
        parser.parse_param(fl_param, argv[1]);
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
    dataset.load_from_file(model_param.path, fl_param);
    DataSet test_dataset;
    bool use_global_test_set = !model_param.test_path.empty();
    if(use_global_test_set)
        test_dataset.load_from_file(model_param.test_path, fl_param);
    Partition partition;
    vector<DataSet> subsets(fl_param.n_parties);
    std::map<int, vector<int>> batch_idxs;

    if(model_param.objective.find("multi:") != std::string::npos || model_param.objective.find("binary:") != std::string::npos) {
        int num_class = dataset.label.size();
        if ((model_param.num_class == 1) && (model_param.num_class != num_class)) {
            LOG(INFO) << "updating number of classes from " << model_param.num_class << " to " << num_class;
            model_param.num_class = num_class;
        }
        if(model_param.num_class > 2)
            model_param.tree_per_rounds = model_param.num_class;
    }
    else if(model_param.objective.find("reg:") != std::string::npos){
        model_param.num_class = 1;
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
