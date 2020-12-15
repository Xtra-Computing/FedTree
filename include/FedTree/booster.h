//
// Created by liqinbin on 12/11/20.
//

#ifndef FEDTREE_BOOSTER_H
#define FEDTREE_BOOSTER_H

#include <FedTree/objective/objective_function.h>
#include <FedTree/metric/metric.h>
// function_builder
#include <FedTree/Tree/tree_builder.h>
#include <FedTree/util/multi_device.h>
#include "FedTree/common.h"
#include "syncarray.h"
#include "FedTree/Tree/tree.h"

//#include "row_sampler.h"

std::mutex mtx;

class Booster {
public:
    void init(const DataSet &dataSet, const GBMParam &param);

    void boost(vector<vector<Tree>> &boosted_model);

private:
    SyncArray<GHPair> gradients;
    std::unique_ptr<ObjectiveFunction> obj;
    std::unique_ptr<Metric> metric;
    SyncArray<float_type> y;
    std::unique_ptr<FunctionBuilder> fbuilder;
//    RowSampler rowSampler;
    GBMParam param;
    int n_devices;
};



#endif //FEDTREE_BOOSTER_H
