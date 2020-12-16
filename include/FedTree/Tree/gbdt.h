//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_GBDT_H
#define FEDTREE_GBDT_H

//Todo: the GBDT model, train a tree, update gradients
#include "tree.h"
#include "FedTree/dataset.h"

class GBDT {
    vector<vector<Tree>> trees;

    GBDT() = default;

    GBDT(const vector<vector<Tree>> gbdt){
        trees = gbdt;
    }

    void train(GBDTParam &param, const DataSet &dataset);

    vector<float_type> predict(const GBDTParam &model_param, const DataSet &dataSet);

    void predict_raw(const GBDTParam &model_param, const DataSet &dataSet, SyncArray<float_type> &y_predict);
};

#endif //FEDTREE_GBDT_H
