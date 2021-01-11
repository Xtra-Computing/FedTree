//
// Created by liqinbin on 10/13/20.
//
#include "FedTree/common.h"
#include "FedTree/dataset.h"

#ifndef FEDTREE_PARTITION_H
#define FEDTREE_PARTITION_H

class Partition {

public:
    std::map<int, vector<int>> homo_partition(const DataSet &dataset, const int n_parties, const bool is_horizontal);

    void hetero_partition(const DataSet &dataset, const int n_parties, const bool is_horizontal, vector<DataSet> &subsets,
                     const vector<float> alpha = {});

    void hybrid_partition(const DataSet &dataset, const int n_parties, vector<float> alpha,
                                     vector<DataSet> &subsets);

};

#endif //FEDTREE_PARTITION_H
