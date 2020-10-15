//
// Created by liqinbin on 10/14/20.
//
#include "FedTree/FL/partition.h"

std::map<int, vector<int>> Partition::homo_partition(const DataSet &dataset, const int n_parties) {
    int n = dataset.n_instances();
    vector<int> idxs;
    for(int i = 0; i < n; i++) {
        idxs.push_back(i);
    }

    std::random_shuffle(idxs.begin(), idxs.end() );

    std::map<int, vector<int>> batch_idxs;

    int stride = n / n_parties;
    for(int i = 0; i < n_parties; i ++) {
        batch_idxs[i] = vector<int>(idxs.begin() + i * stride,
                                    std::min(idxs.end(), idxs.begin() + (i + 1) * stride));
    }
    for(int i = 0; i < n % n_parties; i++) {
        batch_idxs[i].push_back(idxs[n_parties * stride + i]);
    }
    return batch_idxs;
}


