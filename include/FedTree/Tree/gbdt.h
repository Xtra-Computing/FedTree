//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_GBDT_H
#define FEDTREE_GBDT_H

#include "tree.h"
#include "FedTree/dataset.h"

class GBDT {
public:
    vector<vector<Tree>> trees;

    GBDT() = default;

    GBDT(const vector<vector<Tree>> gbdt){
        trees = gbdt;
    }

    void train(GBDTParam &param, DataSet &dataset);

    vector<float_type> predict(const GBDTParam &model_param, const DataSet &dataSet);

    void predict_raw(const GBDTParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict);

    void predict_raw_vertical(const GBDTParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict, std::map<int, vector<int>> &batch_idxs);



    float_type predict_score(const GBDTParam &model_param, const DataSet &dataSet);

    float_type predict_score_vertical(const GBDTParam &model_param, const DataSet &dataSet, std::map<int, vector<int>> &batch_idxs);
};

#endif //FEDTREE_GBDT_H
