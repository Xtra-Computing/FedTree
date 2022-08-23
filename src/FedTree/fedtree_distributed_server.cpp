//
// Created by Yuxuan Han on 11/4/21.
//

#include "FedTree/FL/distributed_server.h"
#include "FedTree/FL/partition.h"
#include "FedTree/parser.h"
#include "FedTree/DP/differential_privacy.h"
#include <sstream>
#include <cmath>


#ifdef _WIN32
INITIALIZE_EASYLOGGINGPP
#endif
int main(int argc, char **argv) {
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);
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


    GBDTParam &param = fl_param.gbdt_param;

    DistributedServer server;
    
    int n_parties = fl_param.n_parties;
    if (fl_param.mode == "vertical") {
        DataSet dataset;
        dataset.load_from_file(model_param.path, fl_param);
        server.VerticalInitVectors(n_parties);
        server.distributed_vertical_init(fl_param, dataset.n_instances(), dataset.y, dataset.label);
    }
    else if (fl_param.mode == "horizontal") {
        server.HorizontalInitVectors(n_parties);
//        int stride = dataset.n_instances() / n_parties;
//        vector<int> n_instances_per_party(n_parties);
//        for (int i = 0; i < n_parties; i++) {
//            n_instances_per_party[i] = stride;
//        }
//        n_instances_per_party[n_parties-1] += dataset.n_instances() - stride * n_parties;
        server.param = fl_param;
//        server.horizontal_init(fl_param, dataset.n_instances(), n_instances_per_party, dataset);
        server.horizontal_init(fl_param);
        server.booster.fbuilder->party_containers_init(fl_param.n_parties);
    }

//    if(param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos) {
//        int num_class = dataset.label.size();
//        if (param.num_class != num_class) {
//            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
//            param.num_class = num_class;
//        }
//        if(param.num_class > 2)
//            param.tree_per_rounds = param.num_class;
//    }
//    else if(param.objective.find("reg:") != std::string::npos){
//        param.num_class = 1;
//    }

    LOG(INFO) << "server init completed.";

    std::string server_address("0.0.0.0:50051");

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&server);
    std::unique_ptr<grpc::Server> grpc_server(builder.BuildAndStart());
    LOG(DEBUG) << "Server listening on " << server_address;
    grpc_server->Wait();
    return 0;
}
