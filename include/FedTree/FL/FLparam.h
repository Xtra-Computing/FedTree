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
    int n_parties;              // number of parties
    bool partition;             // input a single dataset for partitioning or input datasets for each party.
    float alpha;                // the concentration parameter of Dir based partition approaches.
    int n_hori;                 // the number of horizontal partitioning subsets in hybrid partition.
    int n_verti;                // the number of vertical partitioning subsets in hybrid partition.
    string mode;                // "horizontal", "vertical", "hybrid", or "centralized"
    string partition_mode;      // "horizontal", "vertical" or "hybrid"
    string privacy_tech;        // "none" or "he" or "dp"
    float variance;             // variance of dp noise if privacy_tech=="dp"
    GBDTParam gbdt_param;       // parameters for the gbdt training
    float privacy_budget;       // privacy budget param for differential privacy
};


#endif //FEDTREE_FLPARAM_H
