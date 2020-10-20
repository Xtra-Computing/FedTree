//
// Created by liqinbin on 10/13/20.
//

#include "FL/FLparam.h"
#include "common.h"
#include "syncarray.h"

#ifndef FEDTREE_DATASET_H
#define FEDTREE_DATASET_H

// Todo: datset structure (csr and csc). load from file. Refer to ThunderGBM dataset.h https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/dataset.h

class DataSet{
    public:
    ///load dataset from file
    void load_from_file(const string& file_name, FLParam &param);
    void csr_to_csc();

    size_t n_features() const;

    size_t n_instances() const;

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

    // whether the dataset is to big
    int use_cpu = false;
};

#endif //FEDTREE_DATASET_H
