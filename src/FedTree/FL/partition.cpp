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


void Partition::hybrid_partition(const DataSet &dataset, const int n_parties, vector<float> &alpha,
                                 vector<SyncArray<bool>> &feature_map, vector<DataSet> &train_subsets,
                                 int part_length, int part_width){

    for(int i = 0; i < n_parties; i++){
        //todo: group label
        train_subsets[i].n_features_ = dataset.n_features_;
        train_subsets[i].y = dataset.y;
    }
//    int scale_parts = scale * n_parties;
    int ins_interval = dataset.n_instances() / part_width;
    int fea_interval = dataset.n_features() / part_length;
    int n_parts = part_length * part_width;

    int seed = 42;
    std::mt19937 gen(seed);
    dirichlet_distribution<std::mt19937> dir(alpha);
    vector<float> dir_numbers = dir(gen);
    CHECK_EQ(dir_numbers.size(),n_parties);
    vector<int> n_parts_each_party_accu(n_parties + 1, 0);
    for(int i = 1; i < n_parties; i++){
        n_parts_each_party_accu[i] = n_parts_each_party_accu[i-1];
        int n_parts_party = (int) (n_parts * dir_numbers[i-1]);
        if(n_parts_party == 0){
            std::cout<<"one party has no data!"<<std::endl;
            exit(1);
        }
        n_parts_each_party_accu[i] += n_parts_party;
    }
    n_parts_each_party_accu[n_parties] = n_parts;
    vector<int> idx(n_parts);
    thrust::sequence(thrust::host, idx.data(), idx.data()+n_parts, 0);
    std::shuffle(idx.data(), idx.data()+n_parts, gen);
    vector<int> partid2party(n_parts);
//    vector<vector<int>> feature_list (n_parties);
    for(int i = 0; i < n_parties; i++){
        for(int j = n_parts_each_party_accu[i]; j < n_parts_each_party_accu[i+1]; j++){
//            feature_list[i].push_back(j % n_parties);
            partid2party[idx[j]] = i;
        }

    }
    for(int i = 0; i < n_parties; i++) {
        feature_map[i].resize(dataset.n_features());
        auto feature_map_data = feature_map[i].host_data();
        for(int j = 0; j < feature_map[i].size(); j++){
            feature_map_data[j] = false;
        }
    }
    for(int i = 0; i < n_parties; i++){
        train_subsets[i].csr_row_ptr.push_back(0);
    }
    for(int i = 0; i < dataset.csr_row_ptr.size()-1; i++){
        vector<int> train_csr_row_sub(n_parties, 0);
        for(int j = dataset.csr_row_ptr[i]; j < dataset.csr_row_ptr[i+1]; j++){
            float_type value = dataset.csr_val[j];
            int cid = dataset.csr_col_idx[j];
            int part_id = std::min(i / ins_interval, part_width - 1) * part_length +
                          std::min(cid / fea_interval, part_length - 1);
            int party_id = partid2party[part_id];
            feature_map[party_id].host_data()[cid]=true;
            train_subsets[party_id].csr_val.push_back(value);
            train_subsets[party_id].csr_col_idx.push_back(cid);
            train_csr_row_sub[party_id]++;
        }
//        LOG(INFO)<<"3.2";
        for(int i = 0; i < n_parties; i++) {
            train_subsets[i].csr_row_ptr.push_back(train_subsets[i].csr_row_ptr.back()+train_csr_row_sub[i]);
        }
//        LOG(INFO)<<"3.3";
    }
    return;
}


// todo: train_test split. should split the subset to train and test by the sample space.
void Partition::hybrid_partition_with_test(const DataSet &dataset, const int n_parties, vector<float> &alpha,
                                           vector<SyncArray<bool>> &feature_map, vector<DataSet> &train_subsets,
                                           vector<DataSet> &test_subsets, vector<DataSet> &subsets,
                                           int part_length, int part_width, float train_test_fraction){

    for(int i = 0; i < n_parties; i++){
        //todo: group label
        train_subsets[i].n_features_ = dataset.n_features_;
        train_subsets[i].y = dataset.y;
        test_subsets[i].n_features_ = dataset.n_features_;
        test_subsets[i].y = dataset.y;
        subsets[i].n_features_ = dataset.n_features_;
        subsets[i].y = dataset.y;
    }
//    int scale_parts = scale * n_parties;
    int ins_interval = dataset.n_instances() / part_width;
    int fea_interval = dataset.n_features() / part_length;
    int n_parts = part_length * part_width;

    int seed = 42;
    std::mt19937 gen(seed);
    dirichlet_distribution<std::mt19937> dir(alpha);
    vector<float> dir_numbers = dir(gen);
    CHECK_EQ(dir_numbers.size(),n_parties);
    vector<int> n_parts_each_party_accu(n_parties + 1, 0);
    std::cout<<"dir_numbers: "<<dir_numbers[0]<<std::endl;
    for(int i = 1; i < n_parties; i++){
        n_parts_each_party_accu[i] = n_parts_each_party_accu[i-1];
        int n_parts_party = (int) (n_parts * dir_numbers[i-1]);
        if(n_parts_party == 0){
            std::cout<<"one party has no data!"<<std::endl;
            exit(1);
        }
        n_parts_each_party_accu[i] += n_parts_party;
    }
    n_parts_each_party_accu[n_parties] = n_parts;
    vector<int> idx(n_parts);
    thrust::sequence(thrust::host, idx.data(), idx.data()+n_parts, 0);
    std::shuffle(idx.data(), idx.data()+n_parts, gen);
    vector<int> partid2party(n_parts);
    vector<bool> train_or_test(n_parts, true);
//    vector<vector<int>> feature_list (n_parties);
    for(int i = 0; i < n_parties; i++){
        int train_n_parts = (int) ((n_parts_each_party_accu[i+1] - n_parts_each_party_accu[i]) * train_test_fraction);
        for(int j = n_parts_each_party_accu[i]; j < n_parts_each_party_accu[i+1]; j++){
//            feature_list[i].push_back(j % n_parties);
            partid2party[idx[j]] = i;
            if (j >= (n_parts_each_party_accu[i] + train_n_parts))
                train_or_test[idx[j]] = false;
        }

    }
    for(int i = 0; i < n_parties; i++) {
        feature_map[i].resize(dataset.n_features());
        auto feature_map_data = feature_map[i].host_data();
        for(int j = 0; j < feature_map[i].size(); j++){
            feature_map_data[j] = false;
        }
    }
    for(int i = 0; i < n_parties; i++){
        train_subsets[i].csr_row_ptr.push_back(0);
        test_subsets[i].csr_row_ptr.push_back(0);
        subsets[i].csr_row_ptr.push_back(0);
    }
    for(int i = 0; i < dataset.csr_row_ptr.size()-1; i++){
        vector<int> train_csr_row_sub(n_parties, 0);
        vector<int> test_csr_row_sub(n_parties, 0);
        vector<int> csr_row_sub(n_parties, 0);
        for(int j = dataset.csr_row_ptr[i]; j < dataset.csr_row_ptr[i+1]; j++){
            float_type value = dataset.csr_val[j];
            int cid = dataset.csr_col_idx[j];
            int part_id = std::min(i / ins_interval, part_width - 1) * part_length +
                    std::min(cid / fea_interval, part_length - 1);
            int party_id = partid2party[part_id];
            feature_map[party_id].host_data()[cid]=true;
//            LOG(INFO)<<"3.1";
            if(train_or_test[part_id]) {
                train_subsets[party_id].csr_val.push_back(value);
                train_subsets[party_id].csr_col_idx.push_back(cid);
                train_csr_row_sub[party_id]++;
            }
            else{
                test_subsets[party_id].csr_val.push_back(value);
                test_subsets[party_id].csr_col_idx.push_back(cid);
                test_csr_row_sub[party_id]++;
            }
            subsets[party_id].csr_val.push_back(value);
            subsets[party_id].csr_col_idx.push_back(cid);
            csr_row_sub[party_id]++;
        }
//        LOG(INFO)<<"3.2";
        for(int i = 0; i < n_parties; i++) {
            train_subsets[i].csr_row_ptr.push_back(train_subsets[i].csr_row_ptr.back()+train_csr_row_sub[i]);
            test_subsets[i].csr_row_ptr.push_back(test_subsets[i].csr_row_ptr.back()+test_csr_row_sub[i]);
            subsets[i].csr_row_ptr.push_back(subsets[i].csr_row_ptr.back()+csr_row_sub[i]);
        }
//        LOG(INFO)<<"3.3";
    }
    return;
}