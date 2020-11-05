//
// Created by liqinbin on 10/13/20.
//
#include "FedTree/common.h"
#include "FedTree/dataset.h"

#ifndef FEDTREE_PARTITION_H
#define FEDTREE_PARTITION_H
// Todo: different data partitioning strategies. refer to FedKT https://github.com/QinbinLi/FedKT/blob/master/experiments.py line188 partition data

class Partition {

public:
    std::map<int, vector<int>> homo_partition(const DataSet &dataset, const int n_parties, const bool is_horizontal);

    std::map<int, vector<int>>
    hetero_partition(const DataSet &dataset, const int n_parties, const bool is_horizontal,
                     const vector<double> alpha = {});
};

#endif //FEDTREE_PARTITION_H
