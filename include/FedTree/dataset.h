//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_DATASET_H
#define FEDTREE_DATASET_H


#include "FedTree/FL/FLparam.h"
#include "common.h"
#include "syncarray.h"

class DataSet{
    public:
    ///load dataset from file
//    void load_from_file(const string& file_name, FLParam &param);
    void load_from_file(string file_name, FLParam &param);
//    void load_from_file_dense(string file_name, FLParam &param);
    void load_from_files(vector<string>file_names, FLParam &param);
    void load_group_file(string file_name);
    void group_label();
    void load_from_sparse(int n_instances, float *csr_val, int *csr_row_ptr, int *csr_col_idx, float *y,
                                   int *group, int num_group, FLParam &param)
    void load_csc_from_file(string file_name, FLParam &param, int const nfeatures=500);
    void csr_to_csc();

    size_t n_features() const;

    size_t n_instances() const;

//    vector<vector<float_type>> dense_mtx;
    vector<float_type> csr_val;
    vector<int> csr_row_ptr;
    vector<int> csr_col_idx;
    vector<float_type> y;
    size_t n_features_;
    vector<int> group;
    vector<float_type> label;


    // csc variables
    vector<float_type> csc_val;
    vector<int> csc_row_idx;
    vector<int> csc_col_ptr;

    //Todo: SyncArray version
//    SyncArray<float_type> csr_val;
//    SyncArray<int> csr_row_ptr;
//    SyncArray<int> csr_col_idx;
//
//    SyncArray<float_type> csc_val;
//    SyncArray<int> csc_row_idx;
//    SyncArray<int> csc_col_ptr;
    // whether the dataset is to big
    bool use_cpu = true;
    bool has_csc = false;
};

#endif //FEDTREE_DATASET_H
