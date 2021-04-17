//
// Created by Kelly Yung on 2020/11/23.
//

#ifndef FEDTREE_SPLITPOINT_H
#define FEDTREE_SPLITPOINT_H

#include "FedTree/Tree/tree.h"


class SplitPoint {
public:
    float_type gain;
    GHPair fea_missing_gh;//missing gh in this segment
    GHPair rch_sum_gh;//right child total gh (missing gh included if default2right)
    bool default_right;
    int nid;

    //split condition
    int split_fea_id;
    float_type fval;//split on this feature value (for exact)
    unsigned char split_bid;//split on this bin id (for hist)

    bool no_split_value_update; //there is no split value update. Used in build_tree_by_predefined_structure.

    SplitPoint() {
        nid = -1;
        split_fea_id = -1;
        gain = 0;
        no_split_value_update=false;
    }

    SplitPoint(const SplitPoint& copy){
        gain = copy.gain;
        fea_missing_gh.g = copy.fea_missing_gh.g;
        fea_missing_gh.h = copy.fea_missing_gh.h;
        rch_sum_gh.g = copy.rch_sum_gh.g;
        rch_sum_gh.h = copy.rch_sum_gh.h;
        default_right = copy.default_right;
        nid = copy.nid;
        split_fea_id = copy.split_fea_id;
        fval = copy.fval;
        split_bid = copy.split_bid;
        no_split_value_update = copy.no_split_value_update;
    }

    friend std::ostream &operator<<(std::ostream &output, const SplitPoint &sp) {
        output << sp.gain << "/" << sp.split_fea_id << "/" << sp.nid << "/" << sp.rch_sum_gh;
        return output;
    }
};
#endif //FEDTREE_SPLITPOINT_H
