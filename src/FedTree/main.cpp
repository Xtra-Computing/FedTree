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

/*
    //initialize parameters
    FLParam fl_param;
    Parser parser;
    parser.parse_param(fl_param, argc, argv);

    //load dataset from file/files
    DataSet dataset;
    dataset.load_from_file(fl_param.dataset_path);

    //initialize parties and server *with the dataset*
    vector<Party> parties;
    for(i = 0; i < fl_param.n_parties; i++){
        Party party;
        parties.push_back(party);
    }
    Server server;

    //train
    FLtrainer trainer;
    model = trainer.train(parties, server, fl_param);

    //test
    Dataset test_dataset;
    test_dataset.load_from_file(fl_param.test_dataset_path);
    acc = model.predict(test_dataset);
*/

//centralized training test
    FLParam fl_param;
    Parser parser;
    parser.parse_param(fl_param, argc, argv);
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
//    if(fl_param.mode == "centralized") {
//        DataSet dataset;
//        vector <vector<Tree>> boosted_model;
//        dataset.load_from_file(model_param.path, fl_param);
//        GBDT gbdt;
//        gbdt.train(model_param, dataset);
//        parser.save_model("tgbm.model", model_param, gbdt.trees, dataset);
//    }
//    else{
    int n_parties = fl_param.n_parties;
    vector<DataSet> train_subsets(n_parties);
    vector<DataSet> test_subsets(n_parties);
    vector<DataSet> subsets(n_parties);
    vector<SyncArray<bool>> feature_map(n_parties);
    DataSet dataset;
    bool use_global_test_set = !model_param.test_path.empty();
    if(fl_param.partition == true){
        dataset.load_from_file(model_param.path, fl_param);
        Partition partition;
        if(fl_param.partition_mode == "hybrid" or fl_param.partition_mode == "hybrid2"){
            LOG(INFO)<<"horizontal vertical dir";
            if(fl_param.mode == "horizontal")
                CHECK_EQ(fl_param.n_verti, 1);
            else if(fl_param.mode == "vertical")
                CHECK_EQ(fl_param.n_hori, 1);
            if(fl_param.partition_mode == "hybrid")
                partition.horizontal_vertical_dir_partition(dataset, n_parties, fl_param.alpha, feature_map, subsets, fl_param.n_hori, fl_param.n_verti);
            else if(fl_param.partition_mode == "hybrid2")
                partition.hybrid_partition_practical(dataset, n_parties, feature_map, subsets, n_parties, 0.1, 2, 0.1);
	        LOG(INFO)<<"finish partition";
//            std::cout<<"subsets[0].n_instances:"<<subsets[0].n_instances()<<std::endl;
//            std::cout<<"subsets[0].nnz:"<<subsets[0].csr_val.size()<<std::endl;
//            std::cout<<"subsets[1].n_instances:"<<subsets[1].n_instances()<<std::endl;
//            std::cout<<"subsets[1].nnz:"<<subsets[1].csr_val.size()<<std::endl;
//            std::cout<<"subsets[2].n_instances:"<<subsets[2].n_instances()<<std::endl;
//            std::cout<<"subsets[2].nnz:"<<subsets[2].csr_val.size()<<std::endl;
//            std::cout<<"subsets[3].n_instances:"<<subsets[3].n_instances()<<std::endl;
//            std::cout<<"subsets[3].nnz:"<<subsets[3].csr_val.size()<<std::endl;
            if(!use_global_test_set) {
                LOG(INFO) << "train test split";
                for (int i = 0; i < n_parties; i++) {
                    partition.train_test_split(subsets[i], train_subsets[i], test_subsets[i]);
                }
            }
            else{
                for (int i = 0; i < n_parties; i++) {
                    train_subsets[i] = subsets[i];
                }
            }
        }
        else{
            std::cout<<"not supported yet"<<std::endl;
            exit(1);
        }
    }
    else{
        std::cout<<"not supported yet"<<std::endl;
    }

    DataSet test_dataset;
    LOG(INFO)<<"global test";
    if(use_global_test_set)
        test_dataset.load_from_file(model_param.test_path, fl_param);

//    if (ObjectiveFunction::need_group_label(param.gbdt_param.objective)) {
//        group_label();
//        param.gbdt_param.num_class = label.size();
//    }

    GBDTParam &param = fl_param.gbdt_param;
    if(param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos) {
        for(int i = 0; i < n_parties; i++){
            train_subsets[i].group_label();
            test_subsets[i].group_label();
        }
        int num_class = dataset.label.size();
        if (param.num_class != num_class) {
            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
            param.num_class = num_class;
        }
        if(param.num_class > 2)
            param.tree_per_rounds = param.num_class;
    }
    else if(param.objective.find("reg:") != std::string::npos){
        param.num_class = 1;
    }
    
    vector<Party> parties(n_parties);
    vector<int> n_instances_per_party(n_parties);
    LOG(INFO)<<"initialize parties";
    for(int i = 0; i < n_parties; i++){
        parties[i].init(i, train_subsets[i], fl_param, feature_map[i]);
        n_instances_per_party[i] = train_subsets[i].n_instances();
    }
    LOG(INFO)<<"after party init";

    Server server;
    server.init(fl_param, dataset.n_instances(), n_instances_per_party);
    FLtrainer trainer;
    if (param.tree_method == "auto")
        param.tree_method = "hist";
    else if (param.tree_method != "hist"){
        std::cout<<"FedTree only supports histogram-based training yet";
        exit(1);
    }
    std::vector<float_type> scores;
    if(fl_param.mode == "hybrid"){
        LOG(INFO) << "start hybrid trainer";
        trainer.hybrid_fl_trainer(parties, server, fl_param);
        for(int i = 0; i < n_parties; i++){
            float_type score;
            if(use_global_test_set)
                score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_dataset);
            else
                score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_subsets[i]);
            scores.push_back(score);
        }
    }
    else if(fl_param.mode == "ensemble"){
        trainer.ensemble_trainer(parties, server, fl_param);
        float_type score;
        if(use_global_test_set) {
            score = server.global_trees.predict_score(fl_param.gbdt_param, test_dataset);
            scores.push_back(score);
        }
        else
            for(int i = 0; i < n_parties; i++) {
                score = server.global_trees.predict_score(fl_param.gbdt_param, test_subsets[i]);
                scores.push_back(score);
            }
    }
    else if(fl_param.mode == "solo"){
        trainer.solo_trainer(parties, fl_param);
        float_type score;
        for(int i = 0; i < n_parties; i++){
            if(use_global_test_set)
                score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_dataset);
            else
                score = parties[i].gbdt.predict_score(fl_param.gbdt_param, test_subsets[i]);
            scores.push_back(score);
        }
    }
    else if(fl_param.mode == "centralized"){
        GBDT gbdt;
        gbdt.train(fl_param.gbdt_param, dataset);
        float_type score;
        if(use_global_test_set) {
            score = gbdt.predict_score(fl_param.gbdt_param, test_dataset);
            scores.push_back(score);
        }
        else {
            for(int i = 0; i < n_parties; i++) {
                score = gbdt.predict_score(fl_param.gbdt_param, test_subsets[i]);
                scores.push_back(score);
            }
        }
    }
    else if(fl_param.mode == "vertical"){
        trainer.vertical_fl_trainer(parties, server, fl_param);
    }

    float_type mean = 0;
    for(int i = 0; i < scores.size(); i++){
        mean += scores[i];
    }
    mean /= scores.size();
    std::cout<<"mean score:"<<mean<<std::endl;
    float_type var = 0;
    for(int i = 0; i < scores.size(); i++){
        var += (scores[i] - mean) * (scores[i] - mean);
    }
    var /= scores.size();
    float_type std = sqrt(var);
    std::cout<<"std:"<<std<<std::endl;
//        parser.save_model("global_model", fl_param.gbdt_param, server.global_trees.trees, dataset);
//    }
    return 0;
}
