//
// Created by liqinbin on 12/11/20.
//

#ifndef FEDTREE_TRAINER_H
#define FEDTREE_TRAINER_H

#include "common.h"
#include "tree.h"
#include "dataset.h"

class TreeTrainer{
public:
    vector<vector<Tree>> train (GBMParam &param, const Dataset &dataset);
};

#endif //FEDTREE_TRAINER_H
