//
// Created by liqinbin on 12/11/20.
//

#include "FedTree/metric/metric.h"
#include "FedTree/metric/pointwise_metric.h"
#include "FedTree/metric/ranking_metric.h"
#include "FedTree/metric/multiclass_metric.h"

Metric *Metric::create(string name) {
    if (name == "map") return new MAP;
    if (name == "rmse") return new RMSE;
    if (name == "ndcg") return new NDCG;
    if (name == "macc") return new MulticlassAccuracy;
    if (name == "error") return new BinaryClassMetric;
    LOG(FATAL) << "unknown metric " << name;
    return nullptr;
}

void Metric::configure(const GBDTParam &param, const DataSet &dataset) {
    y.resize(dataset.y.size());
    y.copy_from(dataset.y.data(), dataset.n_instances());
}
