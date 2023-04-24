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


void Booster::boost_a_subtree(vector<vector<Tree>> &boosted_model, int n_layer, int *id_list, int *nins_list, 
                                float_type *gradient_g_list, float_type *gradient_h_list, int *n_node, int *nodeid_list, 
                                float_type *input_gradient_g, float_type *input_gradient_h) {
    TIMED_FUNC(timerObj);
    auto gradients_ptr = gradients.host_data();
    for(int i = 0; i < gradients.size(); i++){
        gradients_ptr[i].g = input_gradient_g[i];
        gradients_ptr[i].h = input_gradient_h[i];
    }
    // // update gradients
    // LOG(INFO)<<"input gradient 0"<<gradients_ptr[0];
    // obj->get_gradient(y, fbuilder->get_y_predict(), gradients);
    // LOG(INFO)<<"gradients 0:"<<gradients_ptr[0];
    // for(int i = 0; i < gradients.size(); i++){
    //     if((gradients_ptr[i].g != input_gradient_g[i]) || (gradients_ptr[i].h != input_gradient_h[i])){
    //         LOG(INFO)<<"gradient not equal:"<<i;
    //     }
    //     gradients_ptr[i].g = input_gradient_g[i];
    //     gradients_ptr[i].h = input_gradient_h[i];
    // }
//    if (param.bagging) rowSampler.do_bagging(gradients);
    PERFORMANCE_CHECKPOINT(timerObj);
    //build new model/approximate function
    boosted_model.push_back(fbuilder->build_a_subtree_approximate(gradients, n_layer));
    auto ins2node_id = fbuilder->ins2node_id.host_data();
    Tree &tree = boosted_model[0][0];
    auto tree_node = tree.nodes.host_data();
    vector<int> leaf_list;
    std::map<int, int> leaf_idx_map;
    for(int i = 0; i < tree.nodes.size(); i++){
        if(tree_node[i].is_leaf && tree_node[i].is_valid){
            leaf_idx_map[tree_node[i].final_id] = leaf_list.size();
            nins_list[leaf_list.size()] = tree_node[i].n_instances;
            leaf_list.push_back(tree_node[i].final_id);
            nodeid_list[leaf_list.size()-1] = tree_node[i].final_id;
        }
    }
    *n_node = leaf_list.size();
    vector<vector<int>> id_list_2d(leaf_list.size());
    for(int i = 0; i < y.size(); i++){
        id_list_2d[leaf_idx_map[ins2node_id[i]]].push_back(i);
    }
    auto gradients_data = gradients.host_data();
    int idx = 0;
    for(int i = 0; i < id_list_2d.size(); i++){
        for(int j = 0; j < id_list_2d[i].size(); j++){
            id_list[idx] = id_list_2d[i][j];
            gradient_g_list[idx] = gradients_data[id_list_2d[i][j]].g;
            gradient_h_list[idx] = gradients_data[id_list_2d[i][j]].h;
            idx++;
        }
    }

    PERFORMANCE_CHECKPOINT(timerObj);
    //show metric on training set
    std::ofstream myfile;
    myfile.open ("data.txt", std::ios_base::app);
    myfile << metric->get_score(fbuilder->get_y_predict()) << "\n";
    myfile.close();
    LOG(INFO) << metric->get_name() << " = " << metric->get_score(fbuilder->get_y_predict());
    std::cout<<"after boost a subtree"<<std::endl;
}


void Booster::boost_without_prediction(vector<vector<Tree>> &boosted_model) {
    TIMED_FUNC(timerObj);
    //update gradients
    obj->get_gradient(y, fbuilder->get_y_predict(), gradients);

//    if (param.bagging) rowSampler.do_bagging(gradients);
    PERFORMANCE_CHECKPOINT(timerObj);
    //build new model/approximate function
    boosted_model.push_back(fbuilder->build_approximate(gradients, false));

    PERFORMANCE_CHECKPOINT(timerObj);
    //show metric on training set
    LOG(INFO) << metric->get_name() << " = " << metric->get_score(fbuilder->get_y_predict());
}