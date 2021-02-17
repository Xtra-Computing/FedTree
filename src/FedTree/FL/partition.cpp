//
// Created by liqinbin on 10/14/20.
//
#include "FedTree/FL/partition.h"
#include "FedTree/util/dirichlet.h"
#include "thrust/sequence.h"
#include "thrust/execution_policy.h"
#include "thrust/binary_search.h"
#include <cassert>


void Partition::homo_partition(const DataSet &dataset, const int n_parties, const bool is_horizontal, vector<DataSet> &subsets) {
    int n;
    if (is_horizontal)
        n = dataset.n_instances();
    else
        n = dataset.n_features();

    for(int i = 0; i < n_parties; i++) {
        if(is_horizontal) {
            subsets[i].n_features_ = dataset.n_features();
        }
    }

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

    vector<int> part2party(n);
    for(int i = 0; i < n_parties; i ++) {
        vector<int> part_indexes = batch_idxs[i];
        for(auto& x : part_indexes) {
            part2party[x] = i;
        }
    }

    if(is_horizontal) {
        // TODO: verify whether to modify feature map

        for(int i = 0; i < n_parties; i++) {
            subsets[i].csr_row_ptr.push_back(0);
        }

        for(int i = 0; i < dataset.csr_row_ptr.size()-1; i ++) { // for each row
            vector<int> csr_row_sub(n_parties, 0);
            for(int j = dataset.csr_row_ptr[i]; j < dataset.csr_row_ptr[i+1]; j ++) { // for each element in the row
                float_type value = dataset.csr_val[j];
                int cid = dataset.csr_col_idx[j];
                int part_id = i;
                int party_id = part2party[part_id];

                subsets[party_id].csr_val.push_back(value);
                subsets[party_id].csr_col_idx.push_back(cid);
                csr_row_sub[party_id]++;
            }
            for(int i = 0; i < n_parties; i++) {
                subsets[i].csr_row_ptr.push_back(subsets[i].csr_row_ptr.back() + csr_row_sub[i]);
            }
        }
    } else {
        for(int i = 0; i < n_parties; i++) {
            subsets[i].csc_col_ptr.push_back(0);
        }

        assert(dataset.has_csc);
        // TODO: check the reason why dataset is a const param
//        if(!dataset.has_csc) {
//            dataset.csr_to_csc();
//        }
        for(int i = 0; i < dataset.csc_col_ptr.size()-1; i++) {
            vector<int> csc_col_sub(n_parties, 0);
            for(int j = dataset.csc_col_ptr[i]; j < dataset.csc_col_ptr[i+1]; j ++) {
                float_type value = dataset.csc_val[j];
                int row_id = dataset.csc_row_idx[j];
                int part_id = i;
                int party_id = part2party[part_id];

                subsets[party_id].csc_val.push_back(value);
                subsets[party_id].csc_row_idx.push_back(row_id);
                csc_col_sub[party_id]++;
            }
            for(int i = 0; i < n_parties; i ++) {
                subsets[i].csc_col_ptr.push_back(subsets[i].csc_col_ptr.back() + csc_col_sub[i]);
            }
        }
    }

    return;
}

//Todo add hetero partition according to the labels
void Partition::hetero_partition(const DataSet &dataset, const int n_parties, const bool is_horizontal, vector<DataSet> &subsets,
                            vector<float> alpha) {
    int n;
    if (is_horizontal)
        n = dataset.n_instances();
    else
        n = dataset.n_features();

    // initialize subsets
    for(int i = 0; i < n_parties; i++) {
        if(is_horizontal) {
            subsets[i].n_features_ = dataset.n_features();
        } // n_instances is from y.size(), not a constant
    }

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

    vector<int> part2party(n);
    for (int i = 0; i < n_parties; i++) {
        if (i == 0) {
            vector<int> part_indexes = vector<int>(idxs.begin(), idxs.begin() + int(dirichlet_samples[i]));
            for(int j = 0; j < part_indexes.size(); j ++) {
                int index = part_indexes[j];
                part2party[index] = 0;
            }
        }
            
        else {
            vector<int> part_indexes = vector<int>(idxs.begin() + int(dirichlet_samples[i - 1]),
                                        idxs.begin() + int(dirichlet_samples[i]));
            for(int j = 0; j < part_indexes.size(); j ++) {
                int index = part_indexes[j];
                part2party[index] = i;
            }
        }
    }

    if(is_horizontal) {
        // TODO: verify whether to modify feature map

        for(int i = 0; i < n_parties; i++) {
            subsets[i].csr_row_ptr.push_back(0);
        }

        for(int i = 0; i < dataset.csr_row_ptr.size()-1; i ++) { // for each row
            vector<int> csr_row_sub(n_parties, 0);
            for(int j = dataset.csr_row_ptr[i]; j < dataset.csr_row_ptr[i+1]; j ++) { // for each element in the row
                float_type value = dataset.csr_val[j];
                int cid = dataset.csr_col_idx[j];
                int part_id = i;
                int party_id = part2party[part_id];

                subsets[party_id].csr_val.push_back(value);
                subsets[party_id].csr_col_idx.push_back(cid);
                csr_row_sub[party_id]++;
            }
            for(int i = 0; i < n_parties; i++) {
                subsets[i].csr_row_ptr.push_back(subsets[i].csr_row_ptr.back() + csr_row_sub[i]);
            }
        }
    } else {
        for(int i = 0; i < n_parties; i++) {
            subsets[i].csc_col_ptr.push_back(0);
        }

        assert(dataset.has_csc);
        // TODO: check the reason why dataset is a const param
//        if(!dataset.has_csc) {
//            dataset.csr_to_csc();
//        }
        for(int i = 0; i < dataset.csc_col_ptr.size()-1; i++) {
            vector<int> csc_col_sub(n_parties, 0);
            for(int j = dataset.csc_col_ptr[i]; j < dataset.csc_col_ptr[i+1]; j ++) {
                float_type value = dataset.csc_val[j];
                int row_id = dataset.csc_row_idx[j];
                int part_id = i;
                int party_id = part2party[part_id];

                subsets[party_id].csc_val.push_back(value);
                subsets[party_id].csc_row_idx.push_back(row_id);
                csc_col_sub[party_id]++;
            }
            for(int i = 0; i < n_parties; i ++) {
                subsets[i].csc_col_ptr.push_back(subsets[i].csc_col_ptr.back() + csc_col_sub[i]);
            }
        }
    }

    return;
}


void Partition::hybrid_partition(const DataSet &dataset, const int n_parties, vector<float> &alpha,
                                 vector<SyncArray<bool>> &feature_map, vector<DataSet> &train_subsets,
                                 int part_length, int part_width, int seed){

    for(int i = 0; i < n_parties; i++){
        //todo: group label
        train_subsets[i].n_features_ = dataset.n_features_;
        train_subsets[i].y = dataset.y;
    }
//    int scale_parts = scale * n_parties;
    int ins_interval = dataset.n_instances() / part_width;
    int fea_interval = dataset.n_features() / part_length;
    int n_parts = part_length * part_width;

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

//
//
void Partition::horizontal_vertical_dir_partition(const DataSet &dataset, const int n_parties, float alpha,
                                                  vector<SyncArray<bool>> &feature_map, vector<DataSet> &subsets,
                                                  int n_hori, int n_verti, int seed){
    CHECK_EQ(n_parties, n_hori*n_verti);
    int n_ins = dataset.n_instances();
    std::mt19937 gen(seed);
    vector<int> idxs(n_ins);
    thrust::sequence(thrust::host, idxs.data(), idxs.data()+idxs.size(), 0);
    std::shuffle(idxs.data(), idxs.data()+n_ins, gen);


    vector<float> alpha_vec(n_hori, alpha);
    dirichlet_distribution<std::mt19937> dir(alpha_vec);
    vector<float> dir_numbers = dir(gen);
    LOG(INFO)<<"dir number:"<<dir_numbers;
    CHECK_EQ(dir_numbers.size(), n_hori);
    vector<int> n_ins_each_hori(n_hori + 1);
    n_ins_each_hori[0] = 0;
    for(int i = 1; i < n_hori; i++){
        n_ins_each_hori[i] = n_ins_each_hori[i-1];
        n_ins_each_hori[i] += dir_numbers[i-1] * n_ins;
    }
    n_ins_each_hori[n_hori] = n_ins;

    vector<int> ins2partyid(n_ins);
    #pragma omp parallel for
    for(int i = 0; i < n_hori; i++){
        for(int idx = n_ins_each_hori[i]; idx < n_ins_each_hori[i+1]; idx++){
            ins2partyid[idxs[idx]] = i;
        }
    }

    vector<vector<int>> fea2partyid(n_hori);
    alpha_vec.resize(n_verti, alpha);
    int n_fea = dataset.n_features();
    for(int hori_id = 0; hori_id < n_hori; hori_id++){
        dirichlet_distribution<std::mt19937> dir(alpha_vec);
        vector<float> dir_numbers = dir(gen);
        LOG(INFO)<<"dir_numbers:"<<dir_numbers;
        vector<int> n_fea_each_verti(n_verti + 1);
        n_fea_each_verti[0] = 0;
        for(int i = 1; i < n_verti; i++){
            n_fea_each_verti[i] = n_fea_each_verti[i-1];
            n_fea_each_verti[i] += dir_numbers[i-1] * n_fea;
        }
        n_fea_each_verti[n_verti] = n_fea;
        LOG(INFO)<<"n_fea_each_verti:"<<n_fea_each_verti;
        fea2partyid[hori_id].resize(n_fea);
        vector<int> idxs(n_fea);
        thrust::sequence(thrust::host, idxs.data(), idxs.data()+n_fea, 0);
        std::shuffle(idxs.data(), idxs.data()+n_fea, gen);
        LOG(INFO)<<"idxs:"<<idxs;
        #pragma omp parallel for
        for(int i = 0; i < n_verti; i++){
            for(int idx = n_fea_each_verti[i]; idx < n_fea_each_verti[i+1]; idx++){
                fea2partyid[hori_id][idxs[idx]] = i;
            }
        }
    }
    for(int i = 0; i < n_parties; i++){
        subsets[i].csr_row_ptr.push_back(0);
    }
    for(int i = 0; i < n_parties; i++) {
        // todo: change n_features to the real number of features
        subsets[i].n_features_ = dataset.n_features_;
        feature_map[i].resize(dataset.n_features());
        auto feature_map_data = feature_map[i].host_data();
        for(int j = 0; j < feature_map[i].size(); j++){
            feature_map_data[j] = false;
        }
    }
//    vector<int> y_idx(n_parties, 0);
    for(int i = 0; i < n_ins; i++){
        int hori_id = ins2partyid[i];
        for(int party_id = hori_id * n_verti; party_id < (hori_id + 1) * n_verti; party_id++){
            subsets[party_id].y.push_back(dataset.y[i]);
        }
    }
    for(int i = 0; i < dataset.csr_row_ptr.size()-1; i++){
        vector<int> train_csr_row_sub(n_parties, 0);
        int hori_id = ins2partyid[i];
        for(int j = dataset.csr_row_ptr[i]; j < dataset.csr_row_ptr[i+1]; j++){
            float_type value = dataset.csr_val[j];
            int cid = dataset.csr_col_idx[j];
            int verti_id = fea2partyid[hori_id][cid];
            int party_id = hori_id * n_verti + verti_id;
//            if(party_id == 1)
//                std::cout<<"party_id:"<<party_id<<std::endl;
            feature_map[party_id].host_data()[cid]=true;
            subsets[party_id].csr_val.push_back(value);
            subsets[party_id].csr_col_idx.push_back(cid);
            train_csr_row_sub[party_id]++;
        }
        for(int pid = hori_id * n_verti; pid< (hori_id + 1) * n_verti; pid++){
            subsets[pid].csr_row_ptr.push_back(subsets[pid].csr_row_ptr.back()+train_csr_row_sub[pid]);
        }
//        for(int i = 0; i < n_parties; i++) {
//            if(train_csr_row_sub[i])
//                subsets[i].csr_row_ptr.push_back(subsets[i].csr_row_ptr.back()+train_csr_row_sub[i]);
//        }
    }
}


//1. each party randomly samples some instances with Gau(n_ins/n_party, sigma)
//2. each party randomly samples some features with Gau(n_fea/2(or n_party), sigma)

void Partition::hybrid_partition_practical(const DataSet &dataset, const int n_parties,
                                           vector<SyncArray<bool>> &feature_map, vector<DataSet> &subsets,
                                           int ins_gau_mean_factor, float ins_gau_sigma,
                                           int fea_gau_mean_factor, float fea_gau_sigma, int seed){
    int n_ins = dataset.n_instances();
    int n_fea = dataset.n_features();
    std::mt19937 gen(seed);
    vector<int> ins_idxs(n_ins);
    thrust::sequence(thrust::host, ins_idxs.data(), ins_idxs.data()+ins_idxs.size(), 0);
    vector<int> fea_idxs(n_fea);
    thrust::sequence(thrust::host, fea_idxs.data(), fea_idxs.data()+fea_idxs.size(), 0);
    std::default_random_engine generator(seed);
    std::normal_distribution<double> ins_gau(1.0 / ins_gau_mean_factor, ins_gau_sigma);
    std::normal_distribution<double> fea_gau(1.0 / fea_gau_mean_factor, fea_gau_sigma);
    std::cout<<"test gaussian:";
    for(int i = 0; i < 100; i++){
        std::cout<<ins_gau(generator)<<" ";
    }
    LOG(INFO)<<"1";
    vector<vector<int>> party_ins_idx(n_parties);
    vector<vector<int>> party_fea_idx(n_parties);
    for(int i = 0; i < n_parties; i++){
        subsets[i].csr_row_ptr.push_back(0);
        subsets[i].n_features_ = dataset.n_features_;
        std::shuffle(ins_idxs.data(), ins_idxs.data()+n_ins, gen);
        std::shuffle(fea_idxs.data(), fea_idxs.data()+n_fea, gen);
	    int n_ins_party = (int) (ins_gau(generator) * n_ins);
	    if(n_ins_party > n_ins)
            n_ins_party = n_ins;
        else if(n_ins_party <= 0)
            n_ins_party = 1;
        std::cout<<"n_ins_party:"<<n_ins_party<<std::endl;
        int n_fea_party = (int) (fea_gau(generator) * n_fea);
	    if(n_fea_party > n_fea)
            n_fea_party = n_fea;
        else if(n_fea_party < 0)
            n_fea_party = 1;
        for(int j = 0; j < n_ins_party; j++){
            party_ins_idx[i].push_back(ins_idxs[j]);
        }
        std::cout<<"n_fea_party:"<<n_fea_party<<std::endl;
        LOG(INFO)<<"2";
        feature_map[i].resize(n_fea);
        auto feature_map_data = feature_map[i].host_data();
        for(int j = 0; j < feature_map[i].size(); j++){
            feature_map_data[j] = false;
        }
        for(int j = 0; j < n_fea_party; j++){
            party_fea_idx[i].push_back(fea_idxs[j]);
            feature_map_data[fea_idxs[j]] = true;
        }
    }
    LOG(INFO)<<"3";
    for(int pid = 0; pid < n_parties; pid++){
        thrust::sort(thrust::host, party_ins_idx[pid].data(), party_ins_idx[pid].data()+party_ins_idx[pid].size());
        thrust::sort(thrust::host, party_fea_idx[pid].data(), party_fea_idx[pid].data()+party_fea_idx[pid].size());
    }
    LOG(INFO)<<"4";
    for(int i = 0; i < n_ins; i++) {
        for (int pid = 0; pid < n_parties; pid++) {
            if (thrust::binary_search(thrust::host, party_ins_idx[pid].data(),party_ins_idx[pid].data() + party_ins_idx[pid].size(), i)) {
                subsets[pid].y.push_back(dataset.y[i]);
            }
        }
    }
    LOG(INFO)<<"5";
    for(int i = 0; i < dataset.csr_row_ptr.size()-1; i++){
        vector<int> train_csr_row_sub(n_parties, 0);
        for(int j = dataset.csr_row_ptr[i]; j < dataset.csr_row_ptr[i+1]; j++){
            float_type value = dataset.csr_val[j];
            int cid = dataset.csr_col_idx[j];
            for(int pid = 0; pid < n_parties; pid++){
                if(thrust::binary_search(thrust::host, party_ins_idx[pid].data(), party_ins_idx[pid].data() + party_ins_idx[pid].size(), i)
                &&  thrust::binary_search(thrust::host, party_fea_idx[pid].data(), party_fea_idx[pid].data() + party_fea_idx[pid].size(), cid)){
                    subsets[pid].csr_val.push_back(value);
                    subsets[pid].csr_col_idx.push_back(cid);
                    train_csr_row_sub[pid]++;
                }
            }
        }
        for(int pid = 0; pid < n_parties; pid++){
            if(train_csr_row_sub[pid]){
                subsets[pid].csr_row_ptr.push_back(subsets[pid].csr_row_ptr.back()+train_csr_row_sub[pid]);
            }
        }
    }
}

void Partition::hybrid_partition_with_test(const DataSet &dataset, const int n_parties, vector<float> &alpha,
                                           vector<SyncArray<bool>> &feature_map, vector<DataSet> &train_subsets,
                                           vector<DataSet> &test_subsets, vector<DataSet> &subsets,
                                           int part_length, int part_width, float train_test_fraction, int seed){

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

void Partition::train_test_split(DataSet &dataset, DataSet &train_dataset, DataSet &test_dataset, float train_portion, int split_seed){
    int n_instances = dataset.n_instances();
    int n_train = n_instances * train_portion;
    int n_test = n_instances - n_train;
    vector<int> idxs(n_instances);
    std::cout<<"n_instances:"<<n_instances<<std::endl;
    std::cout<<"dataset.csr_row_ptr.size:"<<dataset.csr_row_ptr.size()<<std::endl;
    CHECK_EQ(dataset.csr_row_ptr.size() - 1, n_instances);
    LOG(INFO)<<"1";
    thrust::sequence(thrust::host, idxs.data(), idxs.data()+n_instances, 0);
    std::mt19937 gen(split_seed);
    std::shuffle(idxs.data(), idxs.data()+n_instances, gen);
    LOG(INFO)<<"train_test_split idxs:"<<idxs;
    vector<bool> idx2train(n_instances, true);
    for(int i = n_train; i < n_instances; i++){
        idx2train[idxs[i]] = false;
    }
    LOG(INFO)<<"2";
    train_dataset.csr_row_ptr.push_back(0);
    test_dataset.csr_row_ptr.push_back(0);
    train_dataset.n_features_ = dataset.n_features_;
    test_dataset.n_features_ = dataset.n_features_;
    int train_idx = 0;
    int test_idx = 0;
    train_dataset.y.resize(n_train);
    test_dataset.y.resize(n_test);
    for(int i = 0; i < n_instances; i++){
        if(idx2train[i]) {
            train_dataset.y[train_idx] = dataset.y[i];
            train_idx++;
        }
        else{
            test_dataset.y[test_idx] = dataset.y[i];
            test_idx++;
        }
    }
    CHECK_EQ(train_idx, n_train);
    CHECK_EQ(test_idx, n_test);
    LOG(INFO)<<"3";

    for(int i = 0; i < dataset.csr_row_ptr.size()-1; i++){
//        int train_csr_row_sub = 0;
//        int test_csr_row_sub = 0;
        for(int j = dataset.csr_row_ptr[i]; j < dataset.csr_row_ptr[i+1]; j++){
            float_type value = dataset.csr_val[j];
            int cid = dataset.csr_col_idx[j];
            if(idx2train[i]){
                train_dataset.csr_val.push_back(value);
                train_dataset.csr_col_idx.push_back(cid);
//                train_csr_row_sub++;
            }
            else{
                test_dataset.csr_val.push_back(value);
                test_dataset.csr_col_idx.push_back(cid);
//                test_csr_row_sub++;
            }
        }
        if(idx2train[i])
            train_dataset.csr_row_ptr.push_back(train_dataset.csr_row_ptr.back() + dataset.csr_row_ptr[i+1] - dataset.csr_row_ptr[i]);
        else
            test_dataset.csr_row_ptr.push_back(test_dataset.csr_row_ptr.back() + dataset.csr_row_ptr[i+1] - dataset.csr_row_ptr[i]);
    }
}
