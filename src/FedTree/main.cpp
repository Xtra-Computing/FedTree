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
        if(fl_param.partition_mode == "hybrid"){
            LOG(INFO)<<"horizontal vertical dir";
            partition.horizontal_vertical_dir_partition(dataset, n_parties, fl_param.alpha, feature_map, subsets, 2, 2);
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
    LOG(INFO)<<"initialize parties";
    for(int i = 0; i < n_parties; i++){
        parties[i].init(i, train_subsets[i], fl_param, feature_map[i]);
    }
    LOG(INFO)<<"after party init";
    Server server;
    server.init(fl_param, dataset.n_instances());
    FLtrainer trainer;
    if (param.tree_method == "auto")
        param.tree_method = "hist";
    else if (param.tree_method != "hist"){
        std::cout<<"FedTree only supports histogram-based training yet";
        exit(1);
    }

    if(fl_param.mode == "hybrid"){
        LOG(INFO) << "start hybrid trainer";
        trainer.hybrid_fl_trainer(parties, server, fl_param);
        for(int i = 0; i < n_parties; i++){
            if(use_global_test_set)
                parties[i].gbdt.predict(fl_param.gbdt_param, test_dataset);
            else
                parties[i].gbdt.predict(fl_param.gbdt_param, test_subsets[i]);
        }
    }
    else if(fl_param.mode == "ensemble"){
        trainer.ensemble_trainer(parties, server, fl_param);
        if(use_global_test_set)
            server.global_trees.predict(fl_param.gbdt_param, test_dataset);
        else
            for(int i = 0; i < n_parties; i++) {
                server.global_trees.predict(fl_param.gbdt_param, test_subsets[i]);
            }
    }
    else if(fl_param.mode == "solo"){
        trainer.solo_trainer(parties, fl_param);
        for(int i = 0; i < n_parties; i++){
            if(use_global_test_set)
                parties[i].gbdt.predict(fl_param.gbdt_param, test_dataset);
            else
                parties[i].gbdt.predict(fl_param.gbdt_param, test_subsets[i]);
        }
    }
    else if(fl_param.mode == "centralized"){
        GBDT gbdt;
        gbdt.train(fl_param.gbdt_param, dataset);
        if(use_global_test_set)
            gbdt.predict(fl_param.gbdt_param, test_dataset);
        else {
            for(int i = 0; i < n_parties; i++) {
                gbdt.predict(fl_param.gbdt_param, test_subsets[i]);
            }
        }
    }

//        parser.save_model("global_model", fl_param.gbdt_param, server.global_trees.trees, dataset);
//    }
    return 0;
}
