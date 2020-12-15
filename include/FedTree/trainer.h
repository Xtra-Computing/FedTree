//
// Created by liqinbin on 12/11/20.
//

#ifndef FEDTREE_TRAINER_H
#define FEDTREE_TRAINER_H

#include "FedTree/common.h"
#include "FedTree/Tree/tree.h"
#include "FedTree/dataset.h"
#include "FedTree/Tree/gbdt.h"





class TreeTrainer{
public:
    vector<vector<Tree>> train (GBMParam &param, const Dataset &dataset);
};

#endif //FEDTREE_TRAINER_H
