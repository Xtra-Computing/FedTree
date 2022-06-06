// created by Tianyuan on 12/1/20

#include "FedTree/Tree/hist_cut.h"
//#include "FedTree/util/device_lambda.h"
//#include "FedTree/util/cub_wrapper.h"
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include "thrust/unique.h"
#include "thrust/execution_policy.h"

#include <algorithm>
#include <random>


void HistCut::get_cut_points_by_data_range(DataSet &dataset, int max_num_bins, int n_instances){
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

#pragma omp parallel for
    for (int fid = 0; fid < n_column; fid++) {
        // copy data value from csc array to unique array
        int col_start = csc_col_ptr_data[fid];
        int col_len = csc_col_ptr_data[fid + 1] - col_start;

        auto val_data = csc_val_data + col_start;
        auto unique_start = unique_vals_data + fid * n_instances;  // notice here

        int unique_len = thrust::unique_copy(thrust::host, val_data, val_data + col_len, unique_start) - unique_start;
        int n_cp = (unique_len <= temp_params_data[1]) ? unique_len : temp_params_data[1];
        temp_row_ptr_data[fid + 1] = unique_len;
        // atomicAdd(&tmp_params_data[0], n_cp);
#pragma omp atomic
        temp_params_data[0] += n_cp;
    }

    // merge the cut points
    temp_params_data = temp_params.host_data();
    cut_points_val.resize(temp_params_data[0]);
    cut_col_ptr.resize(n_column + 1);
    cut_fid.resize(temp_params_data[0]);

    cut_col_ptr.copy_from(temp_row_ptr);
    auto cut_col_ptr_data = cut_col_ptr.host_data();
    temp_row_ptr_data = temp_row_ptr.host_data();
    for(int i = 1; i < (n_column + 1); i++) {
        if(temp_row_ptr_data[i] <= temp_params_data[1])
            cut_col_ptr_data[i] += cut_col_ptr_data[i-1];
        else
            cut_col_ptr_data[i] = cut_col_ptr_data[i-1] + max_num_bins;
    }

    auto cut_point_val_data = cut_points_val.host_data();
    temp_row_ptr_data = temp_row_ptr.host_data();
    cut_col_ptr_data = cut_col_ptr.host_data();
    unique_vals_data = unique_vals.host_data();

#pragma omp parallel for
    for (int fid = 0; fid < n_column; fid++) {
        for (int i = cut_col_ptr_data[fid]; i < cut_col_ptr_data[fid + 1]; i++) {
            int unique_len = temp_row_ptr_data[fid + 1];
            int unique_idx = i - cut_col_ptr_data[fid];
            int cp_idx = (unique_len <= temp_params_data[1]) ? unique_idx : (unique_len / temp_params_data[1] *
                                                                             unique_idx);
            cut_point_val_data[i] = unique_vals_data[fid * n_instances + cp_idx];
        }
    }

    auto cut_fid_data = cut_fid.host_data();
#pragma omp parallel for
    for (int fid = 0; fid < n_column; fid++) {
        for (int i = cut_col_ptr_data[fid]; i < cut_col_ptr_data[fid + 1]; i++) {
            cut_fid_data[i] = fid;
        }
    }
}


template<typename T>
void syncarray_resize_cpu(SyncArray<T> &buf_array, int new_size) {
    CHECK_GE(buf_array.size(), new_size) << "The size of the target Syncarray must greater than the new size. ";
    SyncArray<T> tmp_array(new_size);
    tmp_array.copy_from(buf_array.host_data(), new_size);
    buf_array.resize(new_size);
    buf_array.copy_from(tmp_array);
}

//remove unique values by each flag
void unique_by_flag(SyncArray<float_type> &target_arr, SyncArray<int> &flags, int n_columns) {
    using namespace thrust::placeholders;

//    float max_elem = max_elements(target_arr);
    float max_elem = *thrust::max_element(thrust::host, target_arr.host_data(), target_arr.host_end());
    float min_elem = *thrust::min_element(thrust::host, target_arr.host_data(), target_arr.host_end());
    CHECK_LT(max_elem + n_columns * (max_elem - min_elem + 1), INT_MAX) << "Max_values is too large to be transformed";
    // 1. transform data into unique ranges
    thrust::transform(thrust::host,
                      target_arr.host_data(),
                      target_arr.host_end(),
                      flags.host_data(),
                      target_arr.host_data(),
                      (_1 + _2 * (max_elem - min_elem + 1)));
    // 2. sort the transformed data
    thrust::sort(thrust::host, target_arr.host_data(), target_arr.host_end(), thrust::greater<float>());
    thrust::reverse(thrust::host, flags.host_data(), flags.host_end());
    // 3. eliminate duplicates
    auto new_end = thrust::unique_by_key(thrust::host, target_arr.host_data(), target_arr.host_end(),
                                         flags.host_data());
    int new_size = new_end.first - target_arr.host_data();
    syncarray_resize_cpu(target_arr, new_size);
    syncarray_resize_cpu(flags, new_size);
    // 4. transform data back
    thrust::transform(thrust::host, target_arr.host_data(),
                      target_arr.host_end(),
                      flags.host_data(),
                      target_arr.host_data(),
                      (_1 - _2 * (max_elem - min_elem + 1)));
    thrust::sort_by_key(thrust::host, flags.host_data(), flags.host_end(), target_arr.host_data());
}

// cost more memory
void HistCut::get_cut_points_fast(DataSet &dataset, int max_num_bins, int n_instances) {
//    LOG(INFO) << "Fast getting cut points...";
    if(!dataset.has_csc)
        dataset.csr_to_csc();
    int n_column = dataset.n_features();

    cut_points_val.resize(dataset.csc_val.size());
    cut_col_ptr.resize(dataset.csc_col_ptr.size());
    cut_fid.resize(dataset.csc_val.size());

    cut_points_val.copy_from(&dataset.csc_val[0], dataset.csc_val.size());
    auto csc_ptr = &dataset.csc_col_ptr[0];

    auto cut_fid_data = cut_fid.host_data();
    #pragma omp parallel for
    for(int fid = 0; fid < n_column; fid ++)
        for(int i = csc_ptr[fid]; i < csc_ptr[fid+1]; i++) {
            cut_fid_data[i] = fid;
        }
    unique_by_flag(cut_points_val, cut_fid, n_column);
    //need to reassign the host_data since cut_fid is resized
    cut_fid_data = cut_fid.host_data();
    cut_col_ptr.resize(n_column + 1);
    auto cut_col_ptr_data = cut_col_ptr.host_data();
    for(int fid = 0; fid < cut_fid.size(); fid++){
        *(cut_col_ptr_data + cut_fid_data[fid] + 1) += 1;
    }
    thrust::inclusive_scan(thrust::host, cut_col_ptr_data, cut_col_ptr_data + cut_col_ptr.size(), cut_col_ptr_data);
    SyncArray<int> select_index(cut_fid.size());
    auto select_index_data = select_index.host_data();
    #pragma omp parallel for
    for(int fid = 0; fid < n_column; fid++){
        int interval = (cut_col_ptr_data[fid+1] - cut_col_ptr_data[fid])/max_num_bins;
        for (int i = cut_col_ptr_data[fid]; i < cut_col_ptr_data[fid+1]; i++){
            int feature_idx = i - cut_col_ptr_data[fid];
            if(interval == 0)
                select_index_data[i] = 1;
            else if(feature_idx < max_num_bins)
                select_index_data[cut_col_ptr_data[fid] + interval * feature_idx] = 1;
        }
    }
    auto cut_fid_new_end = thrust::remove_if(thrust::host, cut_fid_data, cut_fid_data+cut_fid.size(), select_index_data,
                                             thrust::not1(thrust::identity<int>()));
    syncarray_resize_cpu(cut_fid, cut_fid_new_end - cut_fid_data);
    auto cut_points_val_data = cut_points_val.host_data();
    auto cut_points_val_new_end = thrust::remove_if(thrust::host, cut_points_val_data, cut_points_val.host_end(),
                                                    select_index_data, thrust::not1(thrust::identity<int>()));
    syncarray_resize_cpu(cut_points_val, cut_points_val_new_end - cut_points_val_data);

    cut_fid_data = cut_fid.host_data();
    cut_col_ptr.resize(n_column + 1);
    cut_col_ptr_data = cut_col_ptr.host_data();
    for(int fid = 0; fid < cut_fid.size(); fid++){
        *(cut_col_ptr_data + cut_fid_data[fid] + 1) += 1;
    }
    thrust::inclusive_scan(thrust::host, cut_col_ptr_data, cut_col_ptr_data + cut_col_ptr.size(), cut_col_ptr_data);

    LOG(DEBUG) << "--->>>>  cut points value: " << cut_points_val;
    LOG(DEBUG) << "--->>>> cut row ptr: " << cut_col_ptr;
    LOG(DEBUG) << "--->>>> cut fid: " << cut_fid;
    LOG(DEBUG) << "TOTAL CP:" << cut_fid.size();
    LOG(DEBUG) << "NNZ: " << dataset.csc_val.size();
}

/**
 * Generate cut points for each feature based on the generated cut points of all parties
 * @param parties_cut
 * @param max_num_bins
 */
void HistCut::get_cut_points_by_parties_cut_sampling(vector<HistCut> &parties_cut, int max_num_bin) {
    // find feature range of each feature for each party
    int n_columns = parties_cut[0].cut_col_ptr.size() - 1;
    vector<vector<float_type>> ranges(n_columns);

    // Merging all cut points into one single cut points
    for (int n = 0; n < n_columns; n++) {
        for (int p = 0; p < parties_cut.size(); p++) {
            auto parties_cut_col_data = parties_cut[p].cut_col_ptr.host_data();
            auto parties_cut_points_val_data = parties_cut[p].cut_points_val.host_data();

            int column_start = parties_cut_col_data[n];
            int column_end = parties_cut_col_data[n+1];

            for (int i = column_start; i < column_end; i++) {
                ranges[n].push_back(parties_cut_points_val_data[i]);
            }
        }
    }
    // Once we have gathered the sorted range, we can randomly sample the cut points to match with the number of bins
    vector<vector<float_type>> samples(n_columns);
    for (int fid = 0; fid < n_columns; fid++) {
        vector<float_type>& sample = samples[fid];

        // Always keep the maximum value
        auto max_element = *std::max_element(ranges[fid].begin(), ranges[fid].end());
        sample.push_back(max_element);

        // Randomly sample number of cut point according to max num bins
        unsigned seed = 0;
        std::shuffle(ranges[fid].begin(), ranges[fid].end(), std::default_random_engine(seed));


        for (int i = 0; i < ranges[fid].size(); i++) {
            if (sample.size() == max_num_bin)
                break;

            auto element = ranges[fid][i];
            // Check if element already in cut points val data
            if (not (std::find(sample.begin(), sample.end(), element) != sample.end()))
                sample.push_back(element);
        }

        // Sort the sample in descending order
        std::sort(sample.begin(), sample.end(), std::greater<float_type>());
    }
    int n_total_bins = 0;
    cut_col_ptr.resize(n_columns + 1);
    auto cut_col_ptr_data = cut_col_ptr.host_data();
    for(int i = 0; i < n_columns; i++){
        n_total_bins += samples[i].size();
        cut_col_ptr_data[i+1] = n_total_bins;
    }
    cut_points_val.resize(n_total_bins);
    cut_fid.resize(n_total_bins);
    auto cut_points_val_data = cut_points_val.host_data();
    auto cut_fid_data = cut_fid.host_data();
    for(int i = 0; i < n_columns; i++){
        for(int j = 0; j < samples[i].size(); j++){
            cut_points_val_data[cut_col_ptr_data[i]+j] = samples[i][j];
            cut_fid_data[cut_col_ptr_data[i]+j] = i;
        }
    }
}

/**
 * Generates cut points for each feature based on numeric ranges of feature values
 * @param f_range Min and max values for each feature.
 * @param max_num_bins Number of cut points for each feature.
 */
void HistCut::get_cut_points_by_feature_range(vector<vector<float_type>> f_range, int max_num_bins) {
    int n_features = f_range.size();
    cut_col_ptr.resize(n_features + 1);
    auto cut_col_ptr_data = cut_col_ptr.host_data();
    vector<int> n_bin_per_features(n_features);
    float inf = std::numeric_limits<float>::infinity();
    #pragma omp parallel for
    for(int fid = 0; fid < n_features; fid ++) {
        float_type val_range = f_range[fid][1] - f_range[fid][0];
        float_type val_step = val_range / max_num_bins;
        int n_bin = max_num_bins;
        if(val_step == 0){
            n_bin = 1;
        }
        else if((val_step == inf) || (val_step == -inf)){
            n_bin = 0;
        }
        else {
            while (val_step < 1e-6) {
                val_step *= 2;
                n_bin /= 2;
            }
            if(n_bin == 0)
                n_bin == 1;
        }
        cut_col_ptr_data[fid + 1] = n_bin;
        n_bin_per_features[fid] = n_bin;
    }
    for(int i = 1; i < cut_col_ptr.size(); i++) {
        cut_col_ptr_data[i] += cut_col_ptr_data[i-1];
    }
    int n_total_bins = cut_col_ptr_data[n_features];
    cut_points_val.resize(n_total_bins);
    cut_fid.resize(n_total_bins);
    auto cut_points_val_data = cut_points_val.host_data();
    auto cut_fid_data = cut_fid.host_data();
    #pragma omp parallel for
    for(int fid = 0; fid < n_features; fid++){
        float_type val_range = f_range[fid][1] - f_range[fid][0];
        int n_bin = n_bin_per_features[fid];
        float_type val_step = val_range / n_bin;
        for(int i = 0; i < n_bin; i ++) {
            cut_fid_data[cut_col_ptr_data[fid] + i] = fid;
            cut_points_val_data[cut_col_ptr_data[fid]+ i] = f_range[fid][1] - i * val_step;
        }
    }



}
