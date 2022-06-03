//
// Created by Kelly Yung on 2020/11/27.
//

#ifndef FEDTREE_OBJECTIVE_FUNCTION_H
#define FEDTREE_OBJECTIVE_FUNCTION_H

#include <FedTree/syncarray.h>
#include <FedTree/dataset.h>

class ObjectiveFunction {
public:
    float constant_h = 0.0;
    virtual void
    get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p, SyncArray<GHPair> &gh_pair) = 0;
    virtual void
    predict_transform(SyncArray<float_type> &y){};
    virtual void configure(GBDTParam param, const DataSet &dataset) {constant_h = param.constant_h;} ;
    virtual string default_metric_name() = 0;

    static ObjectiveFunction* create(string name);

    static bool need_load_group_file(string name);
    static bool need_group_label(string name);
    virtual ~ObjectiveFunction() = default;
};

#endif //FEDTREE_OBJECTIVE_FUNCTION_H
