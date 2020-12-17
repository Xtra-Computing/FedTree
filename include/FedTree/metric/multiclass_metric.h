//
// Created by liqinbin on 12/11/20.
//

#ifndef FEDTREE_MULTICLASS_METRIC_H
#define FEDTREE_MULTICLASS_METRIC_H


#include "FedTree/common.h"
#include "metric.h"

class MulticlassMetric: public Metric {
public:
    void configure(const GBDTParam &param, const DataSet &dataset) override {
        Metric::configure(param, dataset);
        num_class = param.num_class;
        CHECK_EQ(num_class, dataset.label.size());
        label.resize(num_class);
        label.copy_from(dataset.label.data(), num_class);
    }

protected:
    int num_class;
    SyncArray<float_type> label;
};

class MulticlassAccuracy: public MulticlassMetric {
public:
    float_type get_score(const SyncArray<float_type> &y_p) const override;

    string get_name() const override { return "multi-class accuracy"; }
};

class BinaryClassMetric: public MulticlassAccuracy{
public:
    float_type get_score(const SyncArray<float_type> &y_p) const override;
    string get_name() const override { return "test error";}
};


#endif //FEDTREE_MULTICLASS_METRIC_H
