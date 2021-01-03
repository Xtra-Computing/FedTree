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

    std::map<int, vector<int>>
    hetero_partition(const DataSet &dataset, const int n_parties, const bool is_horizontal,
                     const vector<float> alpha = {});

    void hybrid_partition(const DataSet &dataset, const int n_parties, vector<float> &alpha,
                          vector<SyncArray<bool>> &feature_map, vector<DataSet> &subsets,
                          int part_length = 10, int part_width = 10);

    void hybrid_partition_with_test(const DataSet &dataset, const int n_parties, vector<float> &alpha,
                                    vector<SyncArray<bool>> &feature_map, vector<DataSet> &train_subsets,
                                    vector<DataSet> &test_subsets, int part_length=10, int part_width=10,
                                    float train_test_fraction=0.75);

};

#endif //FEDTREE_PARTITION_H
