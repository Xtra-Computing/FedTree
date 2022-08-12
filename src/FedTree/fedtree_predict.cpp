#include "FedTree/FL/FLparam.h"
#include "FedTree/parser.h"
#include "FedTree/dataset.h"
#include "FedTree/predictor.h"
#include "FedTree/Tree/gbdt.h"


#ifdef _WIN32
INITIALIZE_EASYLOGGINGPP
#endif
int main(int argc, char** argv) {
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
    if (model_param.verbose == 0) {
        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "false");
    } else if (model_param.verbose == 1) {
        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
    }

    if (!model_param.profiling) {
        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
    }


    vector<vector<Tree>> boosted_model;
    parser.load_model(model_param.model_path, model_param, boosted_model);
    GBDT gbdt(boosted_model);
    if(model_param.paths.size() > 1){
        CHECK_EQ(fl_param.mode, "vertical");
        CHECK_EQ(fl_param.n_parties, model_param.paths.size());
        vector<DataSet> datasets(fl_param.n_parties);
        for(int i = 0; i < fl_param.n_parties; i++){
            datasets[i].load_from_file(model_param.paths[i], fl_param);
        }
        vector<float_type> y_pred_vec = gbdt.predict(model_param, datasets);
    }
    else {
        DataSet dataset;
        dataset.load_from_file(model_param.path, fl_param);
        vector<float_type> y_pred_vec = gbdt.predict(model_param, dataset);
    }
    
    return 0;
}