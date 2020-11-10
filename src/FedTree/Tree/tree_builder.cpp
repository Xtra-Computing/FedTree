//
// Created by liqinbin on 11/3/20.
//

#include "FedTree/Tree/tree_builder.h"
#include "FedTree/Tree/hist_tree_builder.h"

TreeBuilder *TreeBuilder::create(std::string name) {
    if (name == "hist") return new HistTreeBuilder;
    LOG(FATAL) << "unknown builder " << name;
    return nullptr;
}

SyncArray<GHPair>
HistTreeBuilder::compute_histogram(int n_instances, int n_columns, SyncArray<GHPair> &gradients, HistCut &cut,
                                   SyncArray<unsigned char> &dense_bin_id) {
    auto gh_data = gradients.host_data();
    auto cut_row_ptr_data = cut.cut_row_ptr.host_data();
    int n_bins = n_columns + cut_row_ptr_data[n_columns];
    auto dense_bin_id_data = dense_bin_id.host_data();

    SyncArray<GHPair> hist(n_bins);
    auto hist_data = hist.host_data();

    for (int i = 0; i < n_instances * n_columns; i++) {
        int iid = i / n_columns;
        int fid = i % n_columns;
        unsigned char bid = dense_bin_id_data[iid * n_columns + fid];

        int feature_offset = cut_row_ptr_data[fid] + fid;
        const GHPair src = gh_data[iid];
        GHPair &dest = hist_data[feature_offset + bid];
        if (src.h != 0)
            dest.h += src.h;
        if (src.g != 0)
            dest.g += src.g;
    }

    return hist;
}

SyncArray<GHPair>
HistTreeBuilder::merge_historgrams(MSyncArray<GHPair> &histograms, int n_bins) {

    SyncArray<GHPair> merged_hist(n_bins);
    auto merged_hist_data = merged_hist.host_data();

    for (int i = 0; i < histograms.size(); i++) {
        auto hist_data = histograms[i].host_data();
        for (int j = 0; j < n_bins; j++) {
            GHPair &src = hist_data[j];
            GHPair &dest = merged_hist_data[j];
            if (src.h != 0)
                dest.h += src.h;
            if (src.g != 0)
                dest.g += src.g;
        }
    }

    return merged_hist;
}