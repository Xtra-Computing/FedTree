//
// Created by liqinbin on 10/13/20.
//
#include "FedTree/common.h"
#include "FedTree/dataset.h"

#ifndef FEDTREE_PARTITION_H
#define FEDTREE_PARTITION_H

class Partition {

public:
    void homo_partition(const DataSet &dataset, const int n_parties, const bool is_horizontal, vector<DataSet> &subsets,
                        std::map<int, vector<int>> &batch_idxs, int seed = 42);

    void hetero_partition(const DataSet &dataset, const int n_parties, const bool is_horizontal, vector<DataSet> &subsets,
                     const vector<float> alpha = {});

    void hybrid_partition(const DataSet &dataset, const int n_parties, vector<float> &alpha,
                          vector<SyncArray<bool>> &feature_map, vector<DataSet> &subsets,
                          int part_length = 10, int part_width = 10);

    void hybrid_partition_with_test(const DataSet &dataset, const int n_parties, vector<float> &alpha,
                                    vector<SyncArray<bool>> &feature_map, vector<DataSet> &train_subsets,
                                    vector<DataSet> &test_subsets, vector<DataSet> &subsets,
                                    int part_length=10, int part_width=10, float train_test_fraction=0.75);

    void horizontal_vertical_dir_partition(const DataSet &dataset, const int n_parties, float alpha,
                                           vector<SyncArray<bool>> &feature_map, vector<DataSet> &subsets,
                                           int n_hori = 2, int n_verti = 2);

    void train_test_split(DataSet &dataset, DataSet &train_dataset, DataSet &test_dataset, float train_portion = 0.75);

    std::vector<std::string> split_string_by_delimiter (const string& s, const string& delimiter) {
        size_t pos_start = 0, pos_end, delim_len = delimiter.length();
        string token;
        vector<string> res;

        while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
            token = s.substr (pos_start, pos_end - pos_start);
            pos_start = pos_end + delim_len;
            res.push_back (token);
        }

        res.push_back (s.substr (pos_start));
        return res;
    }

};

#endif //FEDTREE_PARTITION_H
