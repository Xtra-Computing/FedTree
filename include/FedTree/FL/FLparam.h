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
    FLParam() {
        n_parties = 2;
        partition = 1;
        alpha = 100;
        n_hori = 2;
        n_verti = 1;
        mode = "hybrid";
        partition_mode = "hybrid2";
        privacy_tech = "none";
        variance = 0;
        tree_file_path = "../trees.txt";
        scale_gain = 1;
        hybrid_approach = "naive";
    }
    int n_parties; // number of parties
    bool partition; // input a single dataset for partitioning or input datasets for each party.
    float alpha; //the concentration parameter of Dir based partition approaches.
    int n_hori; //the number of horizontal partitioning subsets in hybrid partition.
    int n_verti; //the number of vertical partitioning subsets in hybrid partition.
    string mode; // "horizontal", "vertical", "hybrid", or "centralized"
    string partition_mode; // "horizontal", "vertical" or "hybrid"
    string privacy_tech; //"none" or "he" or "dp"
    float variance; // variance of dp noise if privacy_tech=="dp"
    string tree_file_path; // file path to store the local trees in hybrid fl
    bool scale_gain; // whether scale the gain or not in hybrid fl
    string hybrid_approach;
    GBDTParam gbdt_param; // parameters for the gbdt training
};


#endif //FEDTREE_FLPARAM_H
