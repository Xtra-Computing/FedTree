//
// Created by liqinbin on 12/17/20.
//

#include "FedTree/booster.h"

std::mutex mtx;
void Booster::init(DataSet &dataSet, const GBDTParam &param) {
//    int n_available_device;
//    cudaGetDeviceCount(&n_available_device);
//    CHECK_GE(n_available_device, param.n_device) << "only " << n_available_device
//                                                 << " GPUs available; please set correct number of GPUs to use";
    this->param = param;

    //here
    fbuilder.reset(FunctionBuilder::create(param.tree_method));
    fbuilder->init(dataSet, param);
    obj.reset(ObjectiveFunction::create(param.objective));
    obj->configure(param, dataSet);
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(param, dataSet);

    n_devices = param.n_device;
    int n_outputs = param.num_class * dataSet.n_instances();
    gradients = SyncArray<GHPair>(n_outputs);
    y = SyncArray<float_type>(dataSet.n_instances());
    y.copy_from(dataSet.y.data(), dataSet.n_instances());
}


void Booster::update_gradients() {
    obj->get_gradient(y, fbuilder->get_y_predict(), gradients);
}

void Booster::boost(vector<vector<Tree>> &boosted_model) {
    TIMED_FUNC(timerObj);
    std::unique_lock<std::mutex> lock(mtx);
    //update gradients
    obj->get_gradient(y, fbuilder->get_y_predict(), gradients);

//    if (param.bagging) rowSampler.do_bagging(gradients);
    PERFORMANCE_CHECKPOINT(timerObj);
    //build new model/approximate function
    boosted_model.push_back(fbuilder->build_approximate(gradients));

    PERFORMANCE_CHECKPOINT(timerObj);
    //show metric on training set
    LOG(INFO) << metric->get_name() << " = " << metric->get_score(fbuilder->get_y_predict());
}