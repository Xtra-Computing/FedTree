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
    bool partition; // input a single dataset for partitioning or input datasets for each party.
    float alpha; //the concentration parameter of Dir based partition approaches.
    string mode; // "horizontal", "vertical", "hybrid", or "centralized"
    string privacy_tech; //"none" or "he" or "dp"
    GBDTParam gbdt_param; // parameters for the gbdt training
};


#endif //FEDTREE_FLPARAM_H
