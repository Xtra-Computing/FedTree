// created by Tianyuan on 12/1/20

#include "FedTree/Tree/hist_cut.h"
#include "FedTree/util/device_lambda.h"
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include "thrust/unique.h"
#include "thrust/execution_policy.h"


void HistCut::get_cut_points(DataSet dataset, int max_num_bins, int n_instances){

    int n_column = dataset.n_features();
    SyncArray<float> unique_vals(n_column * n_instances);
    SyncArray<int> temp_row_ptr(n_column + 1);

    SyncArray<int> temp_params(2); //[num_cut_points, max_num_bins]
    int h_temp_params[2] = {0, max_num_bins};
    temp_params.copy_from(h_temp_params, 2);

    auto csc_val_data = &dataset.csc_val[0]; //CPU version; need conversion to syncarray pointer for GPU processing
    auto csc_col_ptr_data = &dataset.csc_col_ptr[0]; //CPU version; need conversion to syncarray pointer for GPU processing
    auto unique_vals_data = unique_vals.host_data();
    auto temp_row_ptr_data = temp_row_ptr.host_data();
    auto temp_params_data = temp_params.host_data();

    for(int fid = 0; fid < n_column; fid ++) {
        int col_start = csc_col_ptr_data[fid];
        int col_len = csc_col_ptr_data[fid+1] - col_start;

        auto val_data = csc_val_data + col_start;
        auto unique_start = unique_vals_data + fid*n_instances;  // notice here
        // TODO: convert to CPU version
        int unique_len = thrust::unique_copy(thrust::device, val_data, val_data + col_len, unique_start) - unique_start;
        int n_cp = (unique_len <= temp_params_data[1]) ? unique_len : temp_params_data[1];
        temp_row_ptr_data[fid+1] = unique_len;
        // atomicAdd(&tmp_params_data[0], n_cp);
        temp_params_data[0] += n_cp;
    }

    // merge the cut points
    temp_params_data = temp_params.host_data();
    cut_points_val.resize(temp_params_data[0]);
    cut_row_ptr.resize(n_column + 1);
    cut_fid.resize(temp_params_data[0]);

    cut_row_ptr.copy_from(temp_row_ptr);
    auto cut_row_ptr_data = cut_row_ptr.host_data();
    temp_row_ptr_data = temp_row_ptr.host_data();
    for(int i = 1; i < (n_column + 1); i++) {
        if(temp_row_ptr_data[i] <= temp_params_data[1])
            cut_row_ptr_data[i] += cut_row_ptr_data[i-1];
        else
            cut_row_ptr_data[i] = cut_row_ptr_data[i-1] + max_num_bins;
    }

    auto cut_point_val_data = cut_points_val.host_data();
    temp_row_ptr_data = temp_row_ptr.host_data();
    cut_row_ptr_data = cut_row_ptr.host_data();
    unique_vals_data = unique_vals.host_data();

    
    for(int fid = 0; fid < n_column; fid ++) {
        for(int i = 0; i < sizeof(cut_row_ptr_data)/sizeof(int); i ++) {
            int unique_len = temp_row_ptr_data[fid+1];
            int unique_idx = i - cut_row_ptr_data[fid];
            int cp_idx = (unique_len <= temp_params_data[1]) ? unique_idx : (unique_len / temp_params_data[1] * unique_idx);
            cut_point_val_data[i] = unique_vals_data[fid*n_instances+cp_idx];
        }
    }

    auto cut_fid_data = cut_fid.host_data();
    for(int fid = 0; fid < n_column; fid ++) {
        for(int i = 0; i < sizeof(cut_row_ptr_data)/sizeof(int); i ++) {
            cut_fid_data[i] = fid;
        }
    }
}
