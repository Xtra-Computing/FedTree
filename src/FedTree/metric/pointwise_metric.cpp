//
// Created by liqinbin on 12/11/20.
//

#include "thrust/reduce.h"
#include "thrust/execution_policy.h"
#include "FedTree/util/device_lambda.h"
#include "FedTree/metric/pointwise_metric.h"

float_type RMSE::get_score(const SyncArray<float_type> &y_p) const {
    CHECK_EQ(y_p.size(), y.size());
    int n_instances = y_p.size();
    SyncArray<float_type> sq_err(n_instances);
    auto sq_err_data = sq_err.host_data();
    const float_type *y_data = y.host_data();
    const float_type *y_predict_data = y_p.host_data();
#pragma omp parallel for
    for (int i = 0; i < n_instances; i++){
        float_type e = y_predict_data[i] - y_data[i];
        sq_err_data[i] = e * e;
    }
    float_type rmse =
            sqrtf(thrust::reduce(thrust::host, sq_err.host_data(), sq_err.host_end()) / n_instances);
    return rmse;
}

