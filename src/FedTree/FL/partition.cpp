//
// Created by liqinbin on 10/14/20.
//
#include "FedTree/FL/partition.h"
#include "FedTree/util/dirichlet.h"
#include <cassert>

std::map<int, vector<int>>
Partition::homo_partition(const DataSet &dataset, const int n_parties, const bool is_horizontal) {
    int n;
    if (is_horizontal)
        n = dataset.n_instances();
    else
        n = dataset.n_features();
    vector<int> idxs;
    for (int i = 0; i < n; i++) {
        idxs.push_back(i);
    }

    std::random_shuffle(idxs.begin(), idxs.end());

    std::map<int, vector<int>> batch_idxs;

    int stride = n / n_parties;
    for (int i = 0; i < n_parties; i++) {
        batch_idxs[i] = vector<int>(idxs.begin() + i * stride,
                                    std::min(idxs.end(), idxs.begin() + (i + 1) * stride));
    }
    for (int i = 0; i < n % n_parties; i++) {
        batch_idxs[i].push_back(idxs[n_parties * stride + i]);
    }
    return batch_idxs;
}

std::map<int, vector<int>>
Partition::hetero_partition(const DataSet &dataset, const int n_parties, const bool is_horizontal,
                            vector<double> alpha) {
    int n;
    if (is_horizontal)
        n = dataset.n_instances();
    else
        n = dataset.n_features();
    vector<int> idxs;
    for (int i = 0; i < n; i++) {
        idxs.push_back(i);
    }
    std::random_shuffle(idxs.begin(), idxs.end());

    if (alpha.empty())
        for (int i = 1; i <= n_parties; i++)
            alpha.push_back(i);
    else
        assert(alpha.size() == n_parties);

    std::random_device rd;
    std::mt19937 gen(rd());
    dirichlet_distribution<std::mt19937> d(alpha);
    vector<double> dirichlet_samples;
    for (double x : d(gen)) dirichlet_samples.push_back(x);
    std::transform(dirichlet_samples.begin(), dirichlet_samples.end(), dirichlet_samples.begin(),
                   [&n](double &c) { return c * n; });
    std::partial_sum(dirichlet_samples.begin(), dirichlet_samples.end(), dirichlet_samples.begin());

    for (auto &x : dirichlet_samples)
        LOG(INFO) << "dirichlet_samples " << x;

    std::map<int, vector<int>> batch_idxs;
    for (int i = 0; i < n_parties; i++) {
        if (i == 0)
            batch_idxs[i] = vector<int>(idxs.begin(), idxs.begin() + int(dirichlet_samples[i]));
        else
            batch_idxs[i] = vector<int>(idxs.begin() + int(dirichlet_samples[i - 1]),
                                        idxs.begin() + int(dirichlet_samples[i]));
    }
    return batch_idxs;
}
