//
// Created by liqinbin on 12/11/20.
//

#ifndef FEDTREE_POINTWISE_METRIC_H
#define FEDTREE_POINTWISE_METRIC_H

#include "metric.h"

class RMSE : public Metric {
public:
    float_type get_score(const SyncArray<float_type> &y_p) const override;

    string get_name() const override { return "RMSE"; }
};

#endif //FEDTREE_POINTWISE_METRIC_H
