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
//split_point[i] stores the split points of feature i

    SyncArray<float_type> cut_points_val;
    SyncArray<int> cut_row_ptr;
    SyncArray<int> cut_fid;

    HistCut() = default;

    HistCut(const HistCut &cut) {
        cut_points_val.copy_from(cut.cut_points_val);
        cut_row_ptr.copy_from(cut.cut_row_ptr);
    }

    // refer to thundergbm histcut.h. Can replace SparseColumns to DataSet.
//    void get_cut_points(SparseColumns &columns, int max_num_bins, int n_instances);

    // equally divide the feature range to get cut points
//    void get_cut_points(float_type feature_min, float_type feature_max, int max_num_bins, int n_instances);
};


#endif //FEDTREE_HIST_CUT_H
