//
// Created by liqinbin on 12/17/20.
//
#include <iostream>
#include <fstream>
#include "FedTree/booster.h"

//std::mutex mtx;

//void Booster::init(const GBDTParam &param, int n_instances) {
//    this -> param = param;
//    fbuilder.reset(new HistTreeBuilder);
//    fbuilder->init(param, n_instances);
//    n_devices = param.n_device;
//    int n_outputs = param.num_class * n_instances;
//    gradients = SyncArray<GHPair>(n_outputs);
//}

void Booster::init(DataSet &dataSet, const GBDTParam &param, bool get_cut_points) {
    this->param = param;
    fbuilder.reset(new HistTreeBuilder);
    if(get_cut_points)
        fbuilder->init(dataSet, param);
    else {
        fbuilder->init_nocutpoints(dataSet, param);
    }
    obj.reset(ObjectiveFunction::create(param.objective));
    obj->configure(param, dataSet);
    if (param.metric == "default") {
        metric.reset(Metric::create(obj->default_metric_name()));
    }else {
        metric.reset(Metric::create(param.metric));
    }
    metric->configure(param, dataSet);
    n_devices = param.n_device;
    int n_outputs = param.num_class * dataSet.n_instances();
    gradients.resize(n_outputs);
    y = SyncArray<float_type>(dataSet.n_instances());
    y.copy_from(dataSet.y.data(), dataSet.n_instances());
}

void Booster::reinit(DataSet &dataSet, const GBDTParam &param){
    //todo: horizontal does not need get_cut_points
    fbuilder->init(dataSet, param);
    int n_outputs = param.num_class * dataSet.n_instances();
    gradients.resize(n_outputs);
    y.resize(dataSet.n_instances());
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

//    auto gradients_data = gradients.host_data();
//    auto gh_data = gh.host_data();
//    for(int i = 0; i < gh.size(); i++)
//        gradients_data[i] = gh_data[i];
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
        DPnoises<float_type>::add_gaussian_noise(gradients_data[i].g, variance);
        DPnoises<float_type>::add_gaussian_noise(gradients_data[i].h, variance);
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
    std::ofstream myfile;
    myfile.open ("data.txt", std::ios_base::app);
    myfile << metric->get_score(fbuilder->get_y_predict()) << "\n";
    myfile.close();
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