//
// Created by Kelly Yung on 2020/12/3. Code taken from ThunderGBM.
//

#ifndef FEDTREE_PREDICTOR_H
#define FEDTREE_PREDICTOR_H

#include "FedTree/Tree/tree.h"
#include <FedTree/dataset.h>

class Predictor{
public:
    void get_y_predict (const GBDTParam &model_param, const vector<vector<Tree>> &boosted_model,
                                   const DataSet &dataSet, SyncArray<float_type> &y_predict);
};

#endif //FEDTREE_PREDICTOR_H