//
// Created by liqinbin on 10/13/20.
//
#include "FedTree/common.h"
#include "FedTree/dataset.h"

#ifndef FEDTREE_PARTITION_H
#define FEDTREE_PARTITION_H

class Partition {

public:
    void homo_partition(const DataSet &dataset, const int n_parties, const bool is_horizontal, vector<DataSet> &subsets);

    void hetero_partition(const DataSet &dataset, const int n_parties, const bool is_horizontal, vector<DataSet> &subsets,
                     const vector<float> alpha = {});

    void hybrid_partition(const DataSet &dataset, const int n_parties, vector<float> &alpha,
                          vector<SyncArray<bool>> &feature_map, vector<DataSet> &subsets,
                          int part_length = 10, int part_width = 10, int seed = 42);

    void hybrid_partition_with_test(const DataSet &dataset, const int n_parties, vector<float> &alpha,
                                    vector<SyncArray<bool>> &feature_map, vector<DataSet> &train_subsets,
                                    vector<DataSet> &test_subsets, vector<DataSet> &subsets,
                                    int part_length=10, int part_width=10, float train_test_fraction=0.75, int seed = 42);

    void horizontal_vertical_dir_partition(const DataSet &dataset, const int n_parties, float alpha,
                                           vector<SyncArray<bool>> &feature_map, vector<DataSet> &subsets,
                                           int n_hori = 2, int n_verti = 2, int seed = 42);

    void train_test_split(DataSet &dataset, DataSet &train_dataset, DataSet &test_dataset, float train_portion = 0.75,
                          int split_seed = 42);
};

#endif //FEDTREE_PARTITION_H
