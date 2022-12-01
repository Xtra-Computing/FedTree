//
// Created by liqinbin on 10/13/20.
//

#include "FedTree/FL/FLparam.h"
#include "FedTree/FL/FLtrainer.h"
#include "FedTree/FL/partition.h"
#include "FedTree/parser.h"
#include "FedTree/dataset.h"
#include "FedTree/Tree/gbdt.h"


#ifdef _WIN32
INITIALIZE_EASYLOGGINGPP
#endif
int main(int argc, char** argv){
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);

//centralized training test
    FLParam fl_param;
    Parser parser;
    if (argc > 1) {
        parser.parse_param(fl_param, argv[1]);
    } else {
        printf("Usage: <config file path> \n");
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
    int n_parties = fl_param.n_parties;

    if (!fl_param.partition || model_param.paths.size() > 1) {
        CHECK_EQ(n_parties, model_param.paths.size());
        fl_param.partition = false;
    }

    vector<DataSet> train_subsets(n_parties);
    vector<DataSet> test_subsets(n_parties);
    vector<DataSet> subsets(n_parties);
    vector<SyncArray<bool>> feature_map(n_parties);
    std::map<int, vector<int>> batch_idxs;
    DataSet dataset;
    bool use_global_test_set = !model_param.test_path.empty();
    if (fl_param.partition == true && fl_param.mode != "centralized") {
        dataset.load_from_file(model_param.path, fl_param);
        Partition partition;
        if (fl_param.partition_mode == "hybrid") {
            LOG(INFO) << "horizontal vertical dir";
            if (fl_param.mode == "horizontal")
                CHECK_EQ(fl_param.n_verti, 1);
            if (fl_param.mode == "vertical")
                CHECK_EQ(fl_param.n_hori, 1);
            partition.horizontal_vertical_dir_partition(dataset, n_parties, fl_param.alpha, feature_map, subsets,
                                                        fl_param.n_hori, fl_param.n_verti);
        } else if (fl_param.partition_mode == "vertical") {
            dataset.csr_to_csc();
            partition.homo_partition(dataset, n_parties, false, subsets, batch_idxs, fl_param.seed);
            for (int i = 0; i < n_parties; i++) {
                train_subsets[i] = subsets[i];
            }
        }else if (fl_param.partition_mode=="horizontal") {
            dataset.csr_to_csc();
            partition.homo_partition(dataset, n_parties, true, subsets, batch_idxs, fl_param.seed);
            for (int i = 0; i < n_parties; i++) {
                train_subsets[i] = subsets[i];
            }
        }
    }
    else if(fl_param.mode != "centralized"){
        for (int i = 0; i < n_parties; i ++) {
            subsets[i].load_from_file(model_param.paths[i], fl_param);
        }
        for (int i = 0; i < n_parties; i++) {
            train_subsets[i] = subsets[i];
        }
    }
    else{
        dataset.load_from_file(model_param.path, fl_param);
    }

    int n_test_dataset = model_param.test_paths.size();
    vector<DataSet> test_dataset(n_test_dataset);
    if (use_global_test_set) {
        if (model_param.reorder_label && fl_param.partition) {
            for (int i = 0; i < n_test_dataset; i++)
                test_dataset[i].label_map = dataset.label_map;
        }
        for (int i = 0; i < n_test_dataset; i++)
            test_dataset[i].load_from_file(model_param.test_paths[i], fl_param);
        if (model_param.reorder_label && fl_param.partition) {
            for (int i = 0; i < n_test_dataset; i++) {
                test_dataset[i].label = dataset.label;
                fl_param.gbdt_param.num_class = test_dataset[i].label.size();
            }
        }
    }

    GBDTParam &param = fl_param.gbdt_param;
    //correct the number of classes
    if (param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos || param.metric == "error") {
        int num_class;
        if(fl_param.partition) {
            num_class = dataset.label.size();
            if ((param.num_class == 1) && (param.num_class != num_class)) {
                LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
                param.num_class = num_class;
            }
        }
        if(param.num_class > 2)
            param.tree_per_round = param.num_class;
    }
    else if(param.objective.find("reg:") != std::string::npos){
        param.num_class = 1;
    }
    vector<Party> parties(n_parties);
    vector<int> n_instances_per_party(n_parties);
    Server server;
    if(fl_param.mode != "centralized") {
        LOG(INFO) << "initialize parties";
        for (int i = 0; i < n_parties; i++) {
            if(fl_param.mode == "vertical")
                parties[i].vertical_init(i, train_subsets[i], fl_param);
            else if(fl_param.mode == "horizontal" || fl_param.mode == "ensemble")
                parties[i].init(i, train_subsets[i], fl_param);
            n_instances_per_party[i] = train_subsets[i].n_instances();
        }
        LOG(INFO) << "initialize server";
        if (fl_param.mode == "vertical") {
            if(fl_param.partition)
                server.vertical_init(fl_param, dataset.n_instances(), n_instances_per_party, dataset.y, dataset.label);
            else
                server.vertical_init(fl_param, train_subsets[0].n_instances(), n_instances_per_party, train_subsets[0].y, train_subsets[0].label);
        } else if (fl_param.mode == "horizontal" || fl_param.mode == "ensemble") {
            server.horizontal_init(fl_param);
        }
    }


    LOG(INFO) << "start training";
    FLtrainer trainer;
    if (param.tree_method == "auto")
        param.tree_method = "hist";
    else if (param.tree_method != "hist"){
        LOG(INFO)<<"FedTree only supports histogram-based training yet";
        exit(1);
    }
    std::vector<float_type> scores;
    if(fl_param.mode == "hybrid"){
        LOG(INFO) << "start hybrid trainer";
        trainer.hybrid_fl_trainer(parties, server, fl_param);
        for(int i = 0; i < n_parties; i++){
            float_type score;
            if(use_global_test_set)
                score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);
            //else
            //    score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_subsets[i]);
            scores.push_back(score);
        }
    }
    else if(fl_param.mode == "ensemble"){
        trainer.ensemble_trainer(parties, server, fl_param);
        float_type score;
        if(use_global_test_set) {
            score = server.global_trees.predict_score(fl_param.gbdt_param, test_dataset[0]);
            scores.push_back(score);
        }
    }
    else if(fl_param.mode == "solo"){
        trainer.solo_trainer(parties, fl_param);
        float_type score;
        for(int i = 0; i < n_parties; i++){
            if(use_global_test_set)
                score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);
            //else
            //    score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_subsets[i]);
            scores.push_back(score);
        }
        float sum = std::accumulate(scores.begin(), scores.end(), 0.0);
        float sq_sum = std::inner_product(scores.begin(), scores.end(), scores.begin(), 0.0);
        float mean = sum / scores.size();
        float std = std::sqrt(sq_sum / scores.size() - mean * mean);
        LOG(INFO)<<"score mean (std):"<< mean << "(" << std << ")";
    }
    else if(fl_param.mode == "centralized"){
        GBDT gbdt;
        gbdt.train(fl_param.gbdt_param, dataset);
        float_type score;
        if(use_global_test_set) {
            score = gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);
            scores.push_back(score);
        }
    } else if (fl_param.mode == "vertical") {
        trainer.vertical_fl_trainer(parties, server, fl_param);
        float_type score;
        if(test_dataset.size() == 1) {
            score = parties[0].gbdt.predict_score_vertical(fl_param.gbdt_param, test_dataset[0], batch_idxs);
        }
        else {
            CHECK_EQ(fl_param.partition, 0);
            score = parties[0].gbdt.predict_score_vertical(fl_param.gbdt_param, test_dataset);
        }
        if(fl_param.joint_prediction)
//            score = server.predict_score_vertical(parties);
            score = 0.0;
        scores.push_back(score);
    }else if (fl_param.mode == "horizontal") {
        LOG(INFO)<<"start horizontal training";
        trainer.horizontal_fl_trainer(parties, server, fl_param);
        LOG(INFO)<<"end horizontal training";
        float_type score;
        if(use_global_test_set)
            score = parties[0].gbdt.predict_score(fl_param.gbdt_param, test_dataset[0]);
        //else
        //    score = parties[0].gbdt.predict_score(fl_param.gbdt_param, test_subsets[0]);
        scores.push_back(score);
    }
    parser.save_model(fl_param.gbdt_param.model_path, fl_param.gbdt_param, server.global_trees.trees);
    return 0;
}
