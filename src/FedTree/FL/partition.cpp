//
// Created by liqinbin on 10/14/20.
//
#include "FedTree/FL/partition.h"
#include "FedTree/util/dirichlet.h"
#include "thrust/sequence.h"
#include "thrust/execution_policy.h"
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

//Todo add hetero partition according to the labels
std::map<int, vector<int>>
Partition::hetero_partition(const DataSet &dataset, const int n_parties, const bool is_horizontal,
                            vector<float> alpha) {
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
    vector<float> dirichlet_samples;
    for (float x : d(gen)) dirichlet_samples.push_back(x);
    std::transform(dirichlet_samples.begin(), dirichlet_samples.end(), dirichlet_samples.begin(),
                   [&n](float &c) { return c * n; });
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


void Partition::hybrid_partition(const DataSet &dataset, const int n_parties, vector<float> alpha,
                                 vector<DataSet> &subsets){
    for(int i = 0; i < n_parties; i++){
        //todo: group label
        subsets[i].n_features_ = dataset.n_features_;
        subsets[i].y = dataset.y;
    }
    int ins_interval = dataset.n_instances() / n_parties;
    int fea_interval = dataset.n_features() / n_parties;
    int n_parts = n_parties * n_parties;

    int seed = 42;
    std::mt19937 gen(seed);
    dirichlet_distribution<std::mt19937> dir(alpha);
    vector<float> dir_numbers = dir(gen);
    CHECK_EQ(dir_numbers.size(),n_parties);
    vector<int> n_parts_each_party_accu(n_parties + 1, 0);
    for(int i = 1; i < n_parties; i++){
        n_parts_each_party_accu[i] = n_parts_each_party_accu[i-1];
        n_parts_each_party_accu[i] += (int) (n_parts * dir_numbers[i]);
    }
    n_parts_each_party_accu[n_parties+1] = n_parts;
    vector<int> idx(n_parts);
    thrust::sequence(thrust::host, idx.data(), idx.data()+n_parts, 0);
    std::shuffle(idx.data(), idx.data()+n_parts, gen);
    vector<int> partid2party(n_parts);
    for(int i = 0; i < n_parties; i++){
        for(int j = n_parts_each_party_accu[i]; j < n_parts_each_party_accu[i+1]; j++){
            partid2party[idx[j]] = i;
        }
    }
//    vector<DataSet> subsets(n_parties);
    for(int i = 0; i < dataset.csr_row_ptr.size()-1; i++){
        vector<int> csr_row_sub(n_parties, 0);
        for(int j = dataset.csr_row_ptr[i]; j < dataset.csr_row_ptr[i+1]; j++){
            float_type value = dataset.csr_val[j];
            int cid = dataset.csr_col_idx[j];
            int part_id = std::min(i / ins_interval, n_parties - 1) * n_parties + std::min(cid / fea_interval, n_parties);
            int party_id = partid2party[part_id];
            subsets[party_id].csr_val.push_back(value);
            subsets[party_id].csr_col_idx.push_back(cid);
            csr_row_sub[party_id]++;
        }
        for(int i = 0; i < n_parties; i++)
            subsets[i].csr_row_ptr.push_back(csr_row_sub[i]);
    }
    return;
}
