//
// Created by liqinbin on 11/3/20.
//

#ifndef FEDTREE_HISTOGRAM_H
#define FEDTREE_HISTOGRAM_H

#include "hist_cut.h"

class Histogram{
    vector<HistCut> cut;
    SyncArray<GHPair> histogram;
};

#endif //FEDTREE_HISTOGRAM_H
