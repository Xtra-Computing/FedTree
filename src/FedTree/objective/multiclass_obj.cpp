//
// Created by liqinbin on 12/15/20.
//


#include "thundergbm/objective/multiclass_obj.h"

void Softmax::configure(GBDTParam param, const DataSet &dataset) {
    num_class = param.num_class;
    label.resize(num_class);
    CHECK_EQ(dataset.label.size(), num_class)<<dataset.label.size() << "!=" << num_class;
    label.copy_from(dataset.label.data(), num_class);
}

void Softmax::get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p, SyncArray<GHPair> &gh_pair) {
    CHECK_EQ(y.size(), y_p.size() / num_class);
    CHECK_EQ(y_p.size(), gh_pair.size());
    auto y_data = y.host_data();
    auto yp_data = y_p.host_data();
    auto gh_data = gh_pair.host_data();
    int num_class = this->num_class;
    int n_instances = y_p.size() / num_class;
#pragma omp parallel for
    for(int i = 0; i < n_instances; i++){
        float_type max = yp_data[i];
        for (int k = 1; k < num_class; ++k) {
            max = fmaxf(max, yp_data[k * n_instances + i]);
        }
        float_type sum = 0;
        for (int k = 0; k < num_class; ++k) {
            //-max to avoid numerical issue
            sum += expf(yp_data[k * n_instances + i] - max);
        }
        for (int k = 0; k < num_class; ++k) {
            float_type p = expf(yp_data[k * n_instances + i] - max) / sum;
            //gradient = p_i - y_i
            //approximate hessian = 2 * p_i * (1 - p_i)
            //https://github.com/dmlc/xgboost/issues/2485
            float_type g = k == y_data[i] ? (p - 1) : (p - 0);
            float_type h = fmaxf(2 * p * (1 - p), 1e-16f);
            gh_data[k * n_instances + i] = GHPair(g, h);
        }
    }
}

void Softmax::predict_transform(SyncArray<float_type> &y) {
    //this method transform y(#class * #instances) into y(#instances)
    auto yp_data = y.host_data();
    auto label_data = label.host_data();
    int num_class = this->num_class == 2 ? 1 : this->num_class;
    int n_instances = y.size() / num_class;
#pragma omp parallel for
    for(int i = 0; i < n_instances; i++){
        int max_k = 0;
        float_type max_p = yp_data[i];
        for (int k = 1; k < num_class; ++k) {
            if (max_p < yp_data[k * n_instances + i]) {
                max_p = yp_data[k * n_instances + i];
                max_k = k;
            }
        }
        yp_data[i] = label_data[max_k];
    }
    //TODO not to make a temp_y?
    SyncArray<float_type> temp_y(n_instances);
    temp_y.copy_from(y.host_data(), n_instances);
    y.resize(n_instances);
    y.copy_from(temp_y);
    LOG(INFO) << "Done y transform size is " << y.size();
}

void SoftmaxProb::predict_transform(SyncArray<float_type> &y) {
    auto yp_data = y.host_data();
    int num_class = this->num_class;
    int n_instances = y.size() / num_class;
#pragma omp parallel for
    for(int i = 0; i < n_instances; i++){
        float_type max = yp_data[i];
        for (int k = 1; k < num_class; ++k) {
            max = fmaxf(max, yp_data[k * n_instances + i]);
        }
        float_type sum = 0;
        for (int k = 0; k < num_class; ++k) {
            //-max to avoid numerical issue
            yp_data[k * n_instances + i] = expf(yp_data[k * n_instances + i] - max);
            sum += yp_data[k * n_instances + i];
        }
        for (int k = 0; k < num_class; ++k) {
            yp_data[k * n_instances + i] /= sum;
        }
    }
}

