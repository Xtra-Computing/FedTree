//
// Created by liqinbin on 11/3/20.
//

#include "FedTree/Tree/tree_builder.h"
#include "FedTree/Tree/hist_tree_builder.h"
#include <limits>
#include <cmath>
#include <algorithm>

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

//assumption: GHPairs in the histograms of all clients are arranged in the same order

SyncArray<GHPair>
HistTreeBuilder::merge_histograms_server_propose(MSyncArray<GHPair> &histograms) {

    int n_bins = histograms[0].size();
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


//assumption 1: bin sizes for the split of a feature are the same
//assumption 2: for each feature, there must be at least 3 bins (2 cut points)
//assumption 3: cut_val_data is sorted by feature id and split value, eg: [f0(0.1), f0(0.2), f0(0.3), f1(100), f1(200),...]
//assumption 4: gradients and hessians are near uniformly distributed

SyncArray<GHPair>
HistTreeBuilder::merge_histograms_client_propose(MSyncArray<GHPair> &histograms, vector<HistCut> &cuts) {
    CHECK_EQ(histograms.size(), cuts.size());
    int n_columns = cuts[0].cut_row_ptr.size() - 1;
    vector<float_type> low(n_columns, std::numeric_limits<float>::max());
    vector<float_type> high(n_columns, -std::numeric_limits<float>::max());
    vector<float_type> resolution(n_columns, std::numeric_limits<float>::max());
    vector<vector<float_type>> bins;
    for (int i = 0; i < cuts.size(); i++) {
        auto cut_val_data = cuts[i].cut_points_val.host_data();
        auto cut_row_ptr_data = cuts[i].cut_row_ptr.host_data();

        for (int j = 0; j < n_columns; j++) {
            int end = cut_row_ptr_data[j + 1];
            int start = cut_row_ptr_data[j];
            float_type res = cut_val_data[end - 1] - cut_val_data[end - 2];
            resolution[j] = std::min(res, resolution[j]);
            float_type l = cut_val_data[start] - res;
            low[j] = std::min(l, low[j]);
            float_type h = cut_val_data[end - 1] + res;
            high[j] = std::max(h, high[j]);

            vector<float_type> v;
            for (float_type f = l; f <= h; f += res)
                v.push_back(f);
            bins.push_back(v);
        }
    }

    int n_bins = 0;
    vector<int> merged_bins_count;
    merged_bins_count.push_back(0);
    for (int i = 0; i < n_columns; i++) {
        float_type count = (high[i] - low[i]) / resolution[i];
        if (abs(int(count) - count) < 1e-5)
            count = int(count);
        else
            count = ceil(count);
        n_bins += count;
        merged_bins_count.push_back(count);
    }

    SyncArray<GHPair> merged_hist(n_bins);
    auto merged_hist_data = merged_hist.host_data();
    for (int i = 0; i < histograms.size(); i++) {
        CHECK_EQ(histograms[i].size(), cuts[i].cut_points_val.size() + n_columns);
        CHECK_EQ(histograms[i].size() + n_columns, bins[i].size());
        auto hist_data = histograms[i].host_data();
        auto cut_row_ptr_data = cuts[i].cut_row_ptr.host_data();
        for (int j = 0; j < n_columns; j++) {
            int client_bin_index_low = cut_row_ptr_data[j] + 2 * j;
            int client_bin_index_high = cut_row_ptr_data[j + 1] + 2 * (j + 1);
            for (int k = 0; k < merged_bins_count[j + 1]; k++) {
                float_type bin_low = low[j] + k * resolution[j];
                float_type bin_high = std::max(low[j] + (k + 1) * resolution[j], high[j]);
                for (int m = client_bin_index_low; m < client_bin_index_high - 1; m++) {
                    if (bin_low < bins[i][m] && bin_high > bins[i][m]) {
                        GHPair &dest = merged_hist_data[merged_bins_count[j] + k];
                        GHPair &src = hist_data[m];
                        float_type factor = (bin_high - bins[i][m]) / (bins[i][m + 1] - bins[i][m]);
                        if (src.h != 0)
                            dest.h += src.h * factor;
                        if (src.g != 0)
                            dest.g += src.g * factor;
                    } else if (bin_low >= bins[i][m] && bin_high <= bins[i][m + 1]) {
                        GHPair &dest = merged_hist_data[merged_bins_count[j] + k];
                        GHPair &src = hist_data[m];
                        float_type factor = (bin_high - bin_low) / (bins[i][m + 1] - bins[i][m]);
                        if (src.h != 0)
                            dest.h += src.h * factor;
                        if (src.g != 0)
                            dest.g += src.g * factor;
                    } else if (bin_high > bins[i][m + 1] && bin_low < bins[i][m + 1]) {
                        GHPair &dest = merged_hist_data[merged_bins_count[j] + k];
                        GHPair &src = hist_data[m];
                        float_type factor = (bins[i][m + 1] - bin_low) / (bins[i][m + 1] - bins[i][m]);
                        if (src.h != 0)
                            dest.h += src.h * factor;
                        if (src.g != 0)
                            dest.g += src.g * factor;
                    }
                }
            }
        }
    }

    return merged_hist;
}
