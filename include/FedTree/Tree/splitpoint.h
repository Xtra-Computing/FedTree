//
// Created by Kelly Yung on 2020/11/23.
//

#ifndef FEDTREE_SPLITPOINT_H
#define FEDTREE_SPLITPOINT_H

#include "FedTree/Tree/tree.h"

class SplitPoint;

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

    SplitPoint() {
        nid = -1;
        split_fea_id = -1;
        gain = 0;
    }

    friend std::ostream &operator<<(std::ostream &output, const SplitPoint &sp) {
        output << sp.gain << "/" << sp.split_fea_id << "/" << sp.nid << "/" << sp.rch_sum_gh;
        return output;
    }
};
#endif //FEDTREE_SPLITPOINT_H
