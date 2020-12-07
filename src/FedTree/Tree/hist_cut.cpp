// created by Tianyuan on 12/1/20

#include "FedTree/Tree/hist_cut.h"

void HistCut::get_cut_points(DataSet dataset, int max_num_bins, int n_instances){
    LOG(INFO) << "Getting cut points";

    int n_column = dataset.n_features();
    SyncArray<float> unique_vals(n_column * n_instances);
    SyncArray<int> temp_row_ptr(n_column + 1);

    SyncArray<int> temp_params(2); //[num_cut_points, max_num_bins]
    int h_temp_params[2] = {0, max_num_bins};
    temp_params.copy_from(h_temp_params, 2);

    auto csc_val_data = dataset.csc_val;
    auto csc_col_ptr_data = dataset.csc_col_ptr;
    auto unique_vals_data = unique_vals.host_data();
    auto temp_row_ptr_data = temp_row_ptr.host_data();
    auto temp_params_data = temp_params.host_data();

    for(int fid = 0; fid < n_column; fid ++) {
        int col_start = csc_col_ptr_data[fid];
        
    }
}
