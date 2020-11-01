//
// Created by hanyuxuan on 28/10/20.
//

#include <algorithm>
#include <numeric>
#include <cassert>
#include "FedTree/Tree/tree_builder.h"

vector<float> TreeBuilder::compute_histogram(vector<float> gradients, vector<int> splits) {
    std::sort(splits.begin(), splits.end());
    assert(splits.front() > 0);
    assert(splits.back() < gradients.size());
    vector<float> histogram;
    histogram.push_back(std::accumulate(gradients.begin(), gradients.begin() + splits[0], 0.0));
    for (int i = 0; i < splits.size() - 1; i++) {
        histogram.push_back(std::accumulate(gradients.begin() + splits[i], gradients.begin() + splits[i + 1], 0.0));
    }
    histogram.push_back(std::accumulate(gradients.begin() + splits.back(), gradients.end(), 0.0));
    return histogram;
}