//
// Created by liqinbin on 12/11/20.
//

#ifndef FEDTREE_BOOSTER_H
#define FEDTREE_BOOSTER_H

#include <FedTree/objective/objective_function.h>
#include <FedTree/metric/metric.h>
// function_builder
#include <FedTree/Tree/tree_builder.h>
#include <FedTree/Tree/hist_tree_builder.h>
#include <FedTree/util/multi_device.h>
#include "FedTree/common.h"
#include "FedTree/syncarray.h"
#include "FedTree/Tree/tree.h"
#include "FedTree/DP/noises.h"


//#include "row_sampler.h"



class Booster {
public:
    void init(DataSet &dataSet, const GBDTParam &param, bool get_cut_points = 1);

    void init (const GBDTParam &param, int n_instances);

    void reinit(DataSet &dataSet, const GBDTParam &param);

    SyncArray<GHPair> get_gradients();

    void set_gradients(SyncArray<GHPair> &gh);

//    void encrypt_gradients(AdditivelyHE::PaillierPublicKey pk);
//
//    void decrypt_gradients(AdditivelyHE::PaillierPrivateKey privateKey);

    void add_noise_to_gradients(float variance);

    void update_gradients();

    void boost(vector<vector<Tree>> &boosted_model);

    void boost_a_subtree(vector<vector<Tree>> &trees, int n_layer, int *id_list, int *nins_list, float *gradient_g_list, 
                        float *gradient_h_list, int *n_node, int *nodeid_list, float *input_gradient_g, float *input_gradient_h);

    void boost_without_prediction(vector<vector<Tree>> &boosted_model);

    GBDTParam param;
    std::unique_ptr<HistTreeBuilder> fbuilder;
    SyncArray<GHPair> gradients;

    std::unique_ptr<Metric> metric;
private:
    int n_devices;
    std::unique_ptr<ObjectiveFunction> obj;
    SyncArray<float_type> y;

    // GBDTParam param;

};




#endif //FEDTREE_BOOSTER_H
