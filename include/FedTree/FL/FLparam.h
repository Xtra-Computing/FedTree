//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_FLPARAM_H
#define FEDTREE_FLPARAM_H

#include "FedTree/Tree/GBDTparam.h"
#include "FedTree/common.h"

// Todo: FLParams structure
class FLParam {
public:
    int n_parties; // number of parties
    string mode; // "horizontal" or "vertical"
    string privacy_tech; //"none" or "he" or "dp"
    GBDTParam GBDTParam; // parameters for the gbdt training
};


#endif //FEDTREE_FLPARAM_H
