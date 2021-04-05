//
// Created by liqinbin on 12/17/20.
//

#include "FedTree/booster.h"

//std::mutex mtx;

void Booster::init(DataSet &dataSet, const GBDTParam &param) {
//    int n_available_device;
//    cudaGetDeviceCount(&n_available_device);
//    CHECK_GE(n_available_device, param.n_device) << "only " << n_available_device
//                                                 << " GPUs available; please set correct number of GPUs to use";
    this->param = param;

//    fbuilder.reset(FunctionBuilder::create(param.tree_method));
    fbuilder.reset(new HistTreeBuilder);
    fbuilder->init(dataSet, param);
    obj.reset(ObjectiveFunction::create(param.objective));
    obj->configure(param, dataSet);
    if (param.metric == "default")
        metric.reset(Metric::create(obj->default_metric_name()));
    else
        metric.reset(Metric::create(param.metric));
    metric->configure(param, dataSet);

    n_devices = param.n_device;
    int n_outputs = param.num_class * dataSet.n_instances();
    gradients = SyncArray<GHPair>(n_outputs);
    y = SyncArray<float_type>(dataSet.n_instances());
    y.copy_from(dataSet.y.data(), dataSet.n_instances());
}

SyncArray<GHPair> Booster::get_gradients() {
    SyncArray<GHPair> gh;
    gh.resize(gradients.size());
    gh.copy_from(gradients);
    return gh;
}

void Booster::set_gradients(SyncArray<GHPair> &gh) {
    gradients.resize(gh.size());
    gradients.copy_from(gh);
}

//void Booster::encrypt_gradients(AdditivelyHE::PaillierPublicKey pk) {
//    auto gradients_data = gradients.host_data();
//    for (int i = 0; i < gradients.size(); i++)
//        gradients_data[i].homo_encrypt(pk);
//}

//void Booster::decrypt_gradients(AdditivelyHE::PaillierPrivateKey privateKey) {
//    auto gradients_data = gradients.host_data();
//    for (int i = 0; i < gradients.size(); i++)
//        gradients_data[i].homo_decrypt(privateKey);
//}

void Booster::add_noise_to_gradients(float variance) {
    auto gradients_data = gradients.host_data();
    for (int i = 0; i < gradients.size(); i++) {
        DPnoises<float>::add_gaussian_noise(gradients_data[i].g, variance);
        DPnoises<float>::add_gaussian_noise(gradients_data[i].h, variance);
    }
}

void Booster::update_gradients() {
    obj->get_gradient(y, fbuilder->get_y_predict(), gradients);
}

void Booster::boost(vector<vector<Tree>> &boosted_model) {
    TIMED_FUNC(timerObj);
//    std::unique_lock<std::mutex> lock(mtx);
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

void Booster::boost_without_prediction(vector<vector<Tree>> &boosted_model) {
    TIMED_FUNC(timerObj);
//    std::unique_lock<std::mutex> lock(mtx);
    //update gradients
    obj->get_gradient(y, fbuilder->get_y_predict(), gradients);
    //LOG(INFO)<<"gradients after updated:"<<gradients;

//    if (param.bagging) rowSampler.do_bagging(gradients);
    PERFORMANCE_CHECKPOINT(timerObj);
    //build new model/approximate function
    boosted_model.push_back(fbuilder->build_approximate(gradients, false));

    PERFORMANCE_CHECKPOINT(timerObj);
    //show metric on training set
    LOG(INFO) << metric->get_name() << " = " << metric->get_score(fbuilder->get_y_predict());
}