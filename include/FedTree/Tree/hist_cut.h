//
// Created by liqinbin on 11/3/20.
//

#ifndef FEDTREE_HIST_CUT_H
#define FEDTREE_HIST_CUT_H

#include "FedTree/common.h"
#include "FedTree/dataset.h"
#include "tree.h"

class HistCut {
public:

    // The vales of cut points
    SyncArray<float_type> cut_points_val;
    // The number of accumulated cut points for current feature
    SyncArray<int> cut_col_ptr;
    // The feature id for current cut point
    SyncArray<int> cut_fid;

    HistCut() = default;

    HistCut(const HistCut &cut) {
        cut_points_val.copy_from(cut.cut_points_val);
        cut_col_ptr.copy_from(cut.cut_col_ptr);
        cut_fid.copy_from(cut.cut_fid);
    }

    // equally divide the feature range to get cut points
    // void get_cut_points(float_type feature_min, float_type feature_max, int max_num_bins, int n_instances);
    void get_cut_points_by_data_range(DataSet &dataset, int max_num_bins, int n_instances);
    void get_cut_points_fast(DataSet &dataset, int max_num_bins, int n_instances);
    void get_cut_points_by_n_instance(DataSet &dataset, int max_num_bins);
    void get_cut_points_by_feature_range(vector<vector<float>> f_range, int max_num_bins);
    void get_cut_points_by_parties_cut_sampling(vector<HistCut> parties_cut);
};


#endif //FEDTREE_HIST_CUT_H
