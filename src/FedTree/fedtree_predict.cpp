#include "FedTree/FL/FLparam.h"
#include "FedTree/parser.h"
#include "FedTree/dataset.h"
#include "FedTree/predictor.h"
#include "FedTree/Tree/gbdt.h"
#include "FedTree/booster.h"


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
    SyncArray<float_type> y_predict;
    if(model_param.paths.size() > 1){
        CHECK_EQ(fl_param.mode, "vertical");
        CHECK_EQ(fl_param.n_parties, model_param.paths.size());
        vector<DataSet> datasets(fl_param.n_parties);
        for(int i = 0; i < fl_param.n_parties; i++){
            datasets[i].load_from_file(model_param.paths[i], fl_param);
        }
        gbdt.predict_raw_vertical(model_param, datasets, y_predict);
        //convert the aggregated values to labels, probabilities or ranking scores.
        std::unique_ptr<ObjectiveFunction> obj;
        obj.reset(ObjectiveFunction::create(model_param.objective));
        obj->configure(model_param, datasets[0]);
        //compute metric
        std::unique_ptr<Metric> metric;
        metric.reset(Metric::create(obj->default_metric_name()));
        metric->configure(model_param, datasets[0]);
        LOG(INFO) << metric->get_name().c_str() << " = " << metric->get_score(y_predict);
    }
    else {
        DataSet dataset;
        dataset.load_from_file(model_param.path, fl_param);
        gbdt.predict_raw(model_param, dataset, y_predict);
        //convert the aggregated values to labels, probabilities or ranking scores.
        std::unique_ptr<ObjectiveFunction> obj;
        obj.reset(ObjectiveFunction::create(model_param.objective));
        obj->configure(model_param, dataset);
        //compute metric
        std::unique_ptr<Metric> metric;
        metric.reset(Metric::create(obj->default_metric_name()));
        metric->configure(model_param, dataset);
        LOG(INFO) << metric->get_name().c_str() << " = " << metric->get_score(y_predict);
    }

    vector<float_type> y_pred_vec(y_predict.size());
    memcpy(y_pred_vec.data(), y_predict.host_data(), sizeof(float_type) * y_predict.size());
    vector<float_type> pred_label;
    if(model_param.objective.find("multi:") != std::string::npos){
        int num_class = model_param.num_class;
        int n_instances = y_pred_vec.size() / num_class;
        pred_label.resize(n_instances);
        SyncArray<int> is_true(n_instances);
        auto is_true_data = is_true.host_data();
    #pragma omp parallel for
        for (int i = 0; i < n_instances; i++){
            int max_k = 0;
            float_type max_p = y_pred_vec[i];
            for (int k = 1; k < num_class; ++k) {
                if (y_pred_vec[k * n_instances + i] > max_p) {
                    max_p = y_pred_vec[k * n_instances + i];
                    max_k = k;
                }
            }
            pred_label[i] = max_k;
        }
    }
    else if(model_param.objective.find("binary:") != std::string::npos){
        std::cout<<"obj binary"<<std::endl;
        pred_label.resize(y_pred_vec.size());
        int n_instances = y_pred_vec.size();
        SyncArray<int> is_true(n_instances);
        auto is_true_data = is_true.host_data();
    #pragma omp parallel for
        for (int i = 0; i < n_instances; i++){
            // change the threshold to 0 if the classes are -1 and 1 and using regression as the objective.
            int max_k = (y_pred_vec[i] > 0.5) ? 1 : 0;
            pred_label[i] = max_k;
        }
    }
    else{
        pred_label = y_pred_vec;
    }

    std::ofstream outfile(fl_param.pred_output);

    if (!outfile) {
        std::cerr << "Unable to open the output file" << std::endl;
        return 1;
    }

    // Write the vector elements to the file
    for (const float_type & element : pred_label) {
        outfile << element << std::endl;
    }

    // Close the output file stream
    outfile.close();
    
    return 0;
}