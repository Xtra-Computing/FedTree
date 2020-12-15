//
// Created by liqinbin on 12/15/20.
//

#ifndef FEDTREE_MULTICLASS_METRIC_H
#define FEDTREE_MULTICLASS_METRIC_H

#include "objective_function.h"
#include "FedTree/util/device_lambda.cuh"

class Softmax : public ObjectiveFunction {
public:
    void get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p,
                      SyncArray<GHPair> &gh_pair) override;

    void predict_transform(SyncArray<float_type> &y) override;

    void configure(GBDTParam param, const DataSet &dataset) override;

    string default_metric_name() override { return "macc"; }

    virtual ~Softmax() override = default;

protected:
    int num_class;
    SyncArray<float_type> label;
};


class SoftmaxProb : public Softmax {
public:
    void predict_transform(SyncArray<float_type> &y) override;

    ~SoftmaxProb() override = default;

};

#endif //FEDTREE_MULTICLASS_METRIC_H
