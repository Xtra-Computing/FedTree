//
// Created by liqinbin on 12/11/20.
//
#include "FedTree/Tree/hist_tree_builder.h"

#include "FedTree/util/cub_wrapper.h"
#include "FedTree/util/device_lambda.h"
#include "thrust/iterator/counting_iterator.h"

#include "thrust/iterator/discard_iterator.h"
#include "thrust/sequence.h"
#include "thrust/binary_search.h"
#include "thrust/execution_policy.h"
#include "FedTree/util/multi_device.h"
#include "FedTree/common.h"
#include <math.h>
#include <algorithm>

#include <math.h>
#include <iterator>
#include <algorithm>
#include <random>


using namespace thrust;

void HistTreeBuilder::init(DataSet &dataset, const GBDTParam &param) {
    TreeBuilder::init(dataset, param);
    if (dataset.n_features_ > 0) {
        cut.get_cut_points_fast(sorted_dataset, param.max_num_bin, n_instances);
        last_hist.resize((2 << param.depth) * cut.cut_points_val.size());
        get_bin_ids();
    }
}

void HistTreeBuilder::init_nocutpoints(DataSet &dataset, const GBDTParam &param) {
    TreeBuilder::init_nosortdataset(dataset, param);
}

SyncArray<GHPair> HistTreeBuilder::get_gradients() {
    SyncArray<GHPair> hist;
    auto &last_hist = this->last_hist;
    hist.copy_from(last_hist);
    return hist;
}

void HistTreeBuilder::set_gradients(SyncArray<GHPair> &gh) {
}


//Tree *HistTreeBuilder::build_tree_level_approximate(int level, int round) {
//    Tree tree;
//    TIMED_FUNC(timerObj);
//    //Todo: add column sampling
//
//    this->ins2node_id.resize(n_instances);
//    this->gradients.set_host_data(const_cast<GHPair *>(gradients.host_data() + round * n_instances));
//    this->trees.init_CPU(this->gradients, param);
//    find_split(level);
////        split_point_all_reduce(level);
//    {
//        TIMED_SCOPE(timerObj, "apply sp");
//        update_tree();
//        update_ins2node_id();
//        {
//            LOG(TRACE) << "gathering ins2node id";
//            //get final result of the reset instance id to node id
//            if (!has_split) {
//                LOG(INFO) << "no splittable nodes, stop";
//                return nullptr;
//            }
//        }
////                ins2node_id_all_reduce(level);
//    }
//
//    return &tree;
//}

void HistTreeBuilder::get_bin_ids() {
//    SparseColumns &columns = shards[device_id].columns;
    HistCut &cut = this->cut;
    auto &dense_bin_id = this->dense_bin_id;
    using namespace thrust;
    int n_column = sorted_dataset.n_features();
    int nnz = sorted_dataset.csc_val.size();
    auto cut_col_ptr = cut.cut_col_ptr.host_data();
    auto cut_points_ptr = cut.cut_points_val.host_data();
    auto csc_val_data = &(sorted_dataset.csc_val[0]);
    auto csc_col_ptr_data = &(sorted_dataset.csc_col_ptr[0]);

    SyncArray<unsigned char> bin_id;
    bin_id.resize(nnz);
    auto bin_id_data = bin_id.host_data();
    int n_block = fminf((nnz / n_column - 1) / 256 + 1, 4 * 56);
    {
        auto lowerBound = [=]__host__(const float_type *search_begin, const float_type *search_end, float_type val) {
            const float_type *left = search_begin;
            const float_type *right = search_end - 1;

            while (left != right) {
                const float_type *mid = left + (right - left) / 2;
                if (*mid <= val)
                    right = mid;
                else left = mid + 1;
            }
            return left;
        };
        TIMED_SCOPE(timerObj, "binning");

#pragma omp parallel for
        for (int cid = 0; cid < n_column; cid++) {
            for (int i = csc_col_ptr_data[cid]; i < csc_col_ptr_data[cid + 1]; i++) {
                auto search_begin = cut_points_ptr + cut_col_ptr[cid];
                auto search_end = cut_points_ptr + cut_col_ptr[cid + 1];
                auto val = csc_val_data[i];
                bin_id_data[i] = lowerBound(search_begin, search_end, val) - search_begin;
            }
        }
    }

    auto max_num_bin = param.max_num_bin;
    dense_bin_id.resize(n_instances * n_column);
    auto dense_bin_id_data = dense_bin_id.host_data();
    auto csc_row_idx_data = sorted_dataset.csc_row_idx.data();
#pragma omp parallel for
    for (int i = 0; i < n_instances * n_column; i++) {
        dense_bin_id_data[i] = max_num_bin;
    }
#pragma omp parallel for
    for (int fid = 0; fid < n_column; fid++) {
        for (int i = csc_col_ptr_data[fid]; i < csc_col_ptr_data[fid + 1]; i++) {
            int row = csc_row_idx_data[i];
            unsigned char bid = bin_id_data[i];
            dense_bin_id_data[row * n_column + fid] = bid;
        }
    }
}


void HistTreeBuilder::find_split(int level) {
    TIMED_FUNC(timerObj);
    std::chrono::high_resolution_clock timer;
    int n_nodes_in_level = 1 << level;
//    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = sorted_dataset.n_features();
    int n_partition = n_column * n_nodes_in_level;
    int n_bins = cut.cut_points_val.size();
    int n_max_nodes = 2 << param.depth;
    int n_max_splits = n_max_nodes * n_bins;

    auto cut_fid_data = cut.cut_fid.host_data();

//    auto i2fid = [=] __host__(int i) { return cut_fid_data[i % n_bins]; };
//    auto hist_fid = make_transform_iterator(counting_iterator<int>(0), i2fid);

    SyncArray<int> hist_fid(n_nodes_in_level * n_bins);
    auto hist_fid_data = hist_fid.host_data();

#pragma omp parallel for
    for (int i = 0; i < hist_fid.size(); i++)
        hist_fid_data[i] = cut_fid_data[i % n_bins];


    int n_split = n_nodes_in_level * n_bins;
    SyncArray<GHPair> missing_gh(n_partition);
    LOG(TRACE) << "start finding split";

    auto t_build_start = timer.now();

    SyncArray<GHPair> hist(n_max_splits);
    SyncArray<float_type> gain(n_max_splits);
    compute_histogram_in_a_level(level, n_max_splits, n_bins, n_nodes_in_level, hist_fid_data, missing_gh, hist);
    //LOG(INFO) << hist;
    compute_gain_in_a_level(gain, n_nodes_in_level, n_bins, hist_fid_data, missing_gh, hist);
    SyncArray<int_float> best_idx_gain(n_nodes_in_level);
    get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins);
    //LOG(INFO) << best_idx_gain;
    get_split_points(best_idx_gain, n_nodes_in_level, hist_fid_data, missing_gh, hist);
    //LOG(INFO) << this->sp;
}


void HistTreeBuilder::find_split_by_predefined_features(int level) {
    TIMED_FUNC(timerObj);
    std::chrono::high_resolution_clock timer;
//    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = sorted_dataset.n_features();
    int n_nodes_in_level = trees.n_nodes_level[level + 1] - trees.n_nodes_level[level];
    int nid_offset = trees.n_nodes_level[level];

    auto nodes_data = trees.nodes.host_data();
    vector<int> n_bins(n_nodes_in_level + 1);
    n_bins[0] = 0;
    auto cut_col_ptr_data = cut.cut_col_ptr.host_data();
    for (int i = 0; i < n_nodes_in_level; i++) {
        n_bins[i + 1] = n_bins[i];
        n_bins[i + 1] += cut_col_ptr_data[nodes_data[nid_offset + i].split_feature_id + 1] -
                         cut_col_ptr_data[nodes_data[nid_offset + i].split_feature_id];
    }
//    int n_max_nodes = 2 << param.depth;
//    int n_split = thrust::reduce(thrust::host, n_bins.data(), n_bins.data() + n_bins.size());
    int n_split = n_bins[n_nodes_in_level];
    sp.resize(n_nodes_in_level);
    auto sp_data = sp.host_data();
    if (n_split == 0) {
        for (int i = 0; i < n_nodes_in_level; i++) {
            sp_data[i].nid = i + nid_offset;
            sp_data[i].no_split_value_update = true;
            sp_data[i].fea_missing_gh = nodes_data[nid_offset + i].sum_gh_pair;
            sp_data[i].rch_sum_gh = 0;
        }
        return;
//        std::cout<<"0 n_split"<<std::endl;
//        exit(0);
    }
    //todo: n_split=0
//    int n_max_splits = n_max_nodes * n_bins;
    auto cut_fid_data = cut.cut_fid.host_data();

    SyncArray<int> hist_fid(n_split);
    //store the split feature id of each possible split
    auto hist_fid_data = hist_fid.host_data();
#pragma omp parallel for
    for (int i = 0; i < n_nodes_in_level; i++)
        for (int j = n_bins[i]; j < n_bins[i + 1]; j++)
            hist_fid_data[j] = nodes_data[nid_offset + i].split_feature_id;

    LOG(TRACE) << "start finding split";

    auto t_build_start = timer.now();
    // todo: thundergbm use n_max_splits size for hist and gain. don't know why. possible reason:
    //  thundergbm last_hist set to maximum possible size (i.e., n_max_splits), to use copy_from, hist is also set the n_max_splits
    SyncArray<GHPair> hist(n_split);

    SyncArray<GHPair> &gh_pair = gradients;
    Tree &tree = trees;
    SyncArray<SplitPoint> &sp = this->sp;

    auto &dense_bin_id = this->dense_bin_id;
    auto &last_hist = this->last_hist;
//    TIMED_FUNC(timerObj);
//    int n_nodes_in_level = static_cast<int>(pow(2, level));
//    int nid_offset = static_cast<int>(pow(2, level) - 1);
//    int n_column = sorted_dataset.n_features();
//    int n_partition = n_column * n_nodes_in_level;
//    int n_bins = cut.cut_points_val.size();
//    int n_max_nodes = 2 << param.depth;
//    int n_max_splits = n_max_nodes * n_bins;
//    int n_split = n_nodes_in_level * n_bins;

    LOG(TRACE) << "start finding split";
    {
        TIMED_SCOPE(timerObj, "build hist");
        if (n_nodes_in_level == 1) {
            auto hist_data = hist.host_data();
            auto cut_col_ptr_data = cut.cut_col_ptr.host_data();
            auto gh_data = gh_pair.host_data();
            auto dense_bin_id_data = dense_bin_id.host_data();
            auto max_num_bin = param.max_num_bin;
            auto n_instances = this->n_instances;
            int split_feature_id = nodes_data[0].split_feature_id;
            CHECK_NE(split_feature_id, -1);
            //has bug if using openmp here. don't know why.
//            #pragma omp parallel for
            for (int i = n_instances * split_feature_id; i < n_instances * (split_feature_id + 1); i++) {
//                int iid = i / n_column;
                int iid = i % n_instances;
                int fid = split_feature_id;
                unsigned char bid = dense_bin_id_data[iid * n_column + fid];
                // todo: check bid when n_bins = 0
                if (n_split == 0) {
                    std::cout << "bid:" << bid;
                }
                if (bid != max_num_bin) {
                    int feature_offset = 0;
                    const GHPair src = gh_data[iid];
                    GHPair &dest = hist_data[feature_offset + bid];
                    if (src.h != 0) {
//                        #pragma omp critical
                        dest.h += src.h;
                    }
                    if (src.g != 0) {
//                        #pragma omp critical
                        dest.g += src.g;
                    }
                }
            }
        } else {
            auto t_dp_begin = timer.now();
//            int n_nodes_in_level = trees.n_nodes_level[level];
            // the instance ids sorted by the node id.
            SyncArray<int> node_idx(n_instances);
            // store the offset of each node in current level in nid4sort
            SyncArray<int> node_ptr(n_nodes_in_level + 1);
            {
                TIMED_SCOPE(timerObj, "data partitioning");
                // the sorted node ids of instances
                SyncArray<int> nid4sort(n_instances);
                nid4sort.copy_from(ins2node_id.host_data(), n_instances);
                sequence(thrust::host, node_idx.host_data(), node_idx.host_end(), 0);
                thrust:
                sort_by_key(thrust::host, nid4sort.host_data(), nid4sort.host_end(), node_idx.host_data());
                auto counting_iter = thrust::make_counting_iterator<int>(nid_offset);
                node_ptr.host_data()[0] =
                        thrust::lower_bound(thrust::host, nid4sort.host_data(), nid4sort.host_end(), nid_offset) -
                        nid4sort.host_data();
                thrust::upper_bound(thrust::host, nid4sort.host_data(), nid4sort.host_end(), counting_iter,
                                    counting_iter + n_nodes_in_level, node_ptr.host_data() + 1);
                LOG(DEBUG) << "node ptr = " << node_ptr;
            }
            auto t_dp_end = timer.now();
            std::chrono::duration<double> dp_used_time = t_dp_end - t_dp_begin;
            this->total_dp_time += dp_used_time.count();


            auto node_ptr_data = node_ptr.host_data();
            auto node_idx_data = node_idx.host_data();
            auto cut_col_ptr_data = cut.cut_col_ptr.host_data();
            auto gh_data = gh_pair.host_data();
            auto dense_bin_id_data = dense_bin_id.host_data();
            auto max_num_bin = param.max_num_bin;
            for (int nid0_to_compute = 0; nid0_to_compute < n_nodes_in_level; ++nid0_to_compute) {

//                int nid0_to_compute = i * 2;
//                int nid0_to_substract = i * 2 + 1;
//                //node_ptr_data[i+1] - node_ptr_data[i] is the number of instances in node i, i is the node id in current level (start from 0)
//                int n_ins_left = node_ptr_data[nid0_to_compute + 1] - node_ptr_data[nid0_to_compute];
//                int n_ins_right = node_ptr_data[nid0_to_substract + 1] - node_ptr_data[nid0_to_substract];
//                if (std::max(n_ins_left, n_ins_right) == 0) continue;
//                //only compute the histogram on the node with the smaller data
//                if (n_ins_left > n_ins_right)
//                    std::swap(nid0_to_compute, nid0_to_substract);
                //compute histogram of leaf and right child
//                for(int  = i*2; nid0_to_compute < i * 2 + 2; nid0_to_compute++)
                {
                    int nid0 = nid0_to_compute;
                    // the start position of instances in nid0
                    auto idx_begin = node_ptr.host_data()[nid0];
                    // the end position of instances in nid0
                    auto idx_end = node_ptr.host_data()[nid0 + 1];
                    auto hist_data = hist.host_data() + n_bins[nid0];
                    this->total_hist_num++;
                    int split_fid = nodes_data[nid_offset + nid0].split_feature_id;
                    //                ThunderGBM: check size of histogram.
                    // todo: zero instance in a node, idx_end - idx_begin = 0. because some previous nodes have no bin, then one child tree have no instance.
                    //has bug if using openmp
//#pragma omp parallel for
                    for (int i = 0; i < idx_end - idx_begin; i++) {
                        int iid = node_idx_data[i + idx_begin];
                        int fid = split_fid;
                        unsigned char bid = dense_bin_id_data[iid * n_column + fid];
                        if (bid != max_num_bin) {
                            int feature_offset = 0;
                            const GHPair src = gh_data[iid];
                            GHPair &dest = hist_data[feature_offset + bid];
                            if (src.h != 0) {
                                //#pragma omp atomic update
                                dest.h += src.h;
                            }
                            if (src.g != 0) {
                                //#pragma omp atomic update
                                dest.g += src.g;
                            }
                        }
                    }
                }
//                //subtract to the histogram of the other node
//                auto t_copy_start = timer.now();
//                {
//                    auto hist_data_computed = hist.host_data() + n_bins[nid0_to_compute];
//                    auto hist_data_to_compute = hist.host_data() + n_bins[nid0_to_substract];
//                    auto father_hist_data = last_hist.host_data() + n_bins[nid0_to_substract/2];
//                    //#pragma omp parallel for
//                    for(int i = 0; i < n_bins; i++){
//                        hist_data_to_compute[i] = father_hist_data[i] - hist_data_computed[i];
//                    }
//                }
//                auto t_copy_end = timer.now();
//                std::chrono::duration<double> cp_used_time = t_copy_end - t_copy_start;
//                this->total_copy_time += cp_used_time.count();
//                            PERFORMANCE_CHECKPOINT(timerObj);
            }  // end for each node
        }
        last_hist.copy_from(hist.host_data(), n_split);
    }
    this->build_n_hist++;
    inclusive_scan_by_key(thrust::host, hist_fid_data, hist_fid_data + n_split,
                          hist.host_data(), hist.host_data());
    LOG(DEBUG) << hist;

    //    int n_partition = n_column * n_nodes_in_level;
    // missing gh for each node. If there is no bin inside a node, then the corresponding missing gh is not assigned.
    SyncArray<GHPair> missing_gh(n_nodes_in_level);
    const auto missing_gh_data = missing_gh.host_data();
    auto cut_col_ptr = cut.cut_col_ptr.host_data();
    auto hist_data = hist.host_data();
#pragma omp parallel for
    for (int nid0 = 0; nid0 < n_nodes_in_level; nid0++) {
        int nid = nid0 + nid_offset;
        if (!nodes_data[nid].splittable()) continue;
        if (n_bins[nid0 + 1] != n_bins[nid0]) {
            GHPair node_gh = hist_data[n_bins[nid0 + 1] - 1];
            missing_gh_data[nid0] = nodes_data[nid].sum_gh_pair - node_gh;
        }
    }
    LOG(DEBUG) << missing_gh;
    auto compute_gain = []__host__(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight,
                                   float_type lambda) -> float_type {
        if (lch.h >= min_child_weight && rch.h >= min_child_weight)
            return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda) -
                   (father.g * father.g) / (father.h + lambda);
        else
            return 0;
    };

    GHPair *gh_prefix_sum_data = hist.host_data();
    //gain for each split
    SyncArray<float_type> gain(n_split);
    float_type *gain_data = gain.host_data();
//    auto ignored_set_data = ignored_set.host_data();
    //for lambda expression
    float_type mcw = param.min_child_weight;
    float_type l = param.lambda;

    // todo: change to one for loop, use binary search to find the node id, compare performance
    for (int nid0 = 0; nid0 < n_nodes_in_level; nid0++) {
        int nid = nid0 + nid_offset;
#pragma omp parallel for
        for (int i = n_bins[nid0]; i < n_bins[nid0 + 1]; i++) {
            int fid = hist_fid_data[i];
            if (nodes_data[nid].is_valid) {
                int pid = nid0;
                GHPair father_gh = nodes_data[nid].sum_gh_pair;
                GHPair p_missing_gh = missing_gh_data[pid];
                GHPair rch_gh = gh_prefix_sum_data[i];
                float_type default_to_left_gain = std::max(0.f,
                                                           compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
                rch_gh = rch_gh + p_missing_gh;
                float_type default_to_right_gain = std::max(0.f,
                                                            compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw,
                                                                         l));
                if (default_to_left_gain > default_to_right_gain)
                    gain_data[i] = default_to_left_gain;
                else
                    gain_data[i] = -default_to_right_gain;//negative means default split to right
            } else gain_data[i] = 0;
        }
    }
    //only have the best idx and gain for the nodes with bins
    SyncArray<int_float> best_idx_gain(n_nodes_in_level);
    {
        TIMED_SCOPE(timerObj, "get best gain");
        auto arg_abs_max = []__host__(const int_float &a, const int_float &b) {
            if (fabsf(thrust::get<1>(a)) == fabsf(thrust::get<1>(b)))
                return thrust::get<0>(a) < thrust::get<0>(b) ? a : b;
            else
                return fabsf(thrust::get<1>(a)) > fabsf(thrust::get<1>(b)) ? a : b;
        };

        SyncArray<int> nid_iterator(n_split);
        auto nid_iterator_data = nid_iterator.host_data();
        for (int nid = 0; nid < n_nodes_in_level; nid++) {
            for (int bid = n_bins[nid]; bid < n_bins[nid + 1]; bid++)
                nid_iterator_data[bid] = nid;
        }

        thrust::reduce_by_key(
                thrust::host,
                nid_iterator_data, nid_iterator_data + n_split,
                make_zip_iterator(make_tuple(thrust::counting_iterator<int>(0), gain.host_data())),
                make_discard_iterator(),
                best_idx_gain.host_data(),
                thrust::equal_to<int>(),
                arg_abs_max
        );
        LOG(DEBUG) << n_split;
        LOG(DEBUG) << "best rank & gain = " << best_idx_gain;
    }
    //note: the size of best_idx_gain may not be equal to n_nodes_in_level, since some nodes may not have bin.
    const int_float *best_idx_gain_data = best_idx_gain.host_data();
    auto cut_val_data = cut.cut_points_val.host_data();


    vector<int> best_idx_gain_idx(n_nodes_in_level);
    int idx = 0;
    for (int i = 0; i < best_idx_gain_idx.size(); i++) {
        if (n_bins[i + 1] == n_bins[i]) {
            best_idx_gain_idx[i] = -1;
            continue;
        }
        best_idx_gain_idx[i] = idx;
        idx++;
    }
#pragma omp parallel for
    for (int i = 0; i < n_nodes_in_level; i++) {
        if (n_bins[i + 1] == n_bins[i]) {
//            sp_data[i].split_bid = nodes_data[i + nid_offset].split_bid;
//            sp_data[i].fval = nodes_data[i + nid_offset].split_value;
//            sp_data[i].rch_sum_gh = nodes_data[i + nid_offset].sum_gh_pair;
//            sp_data[i].split_fea_id = nodes_data[i + nid_offset].split_feature_id;
//            sp_data[i].nid = i + nid_offset;
//            sp_data[i].gain = nodes_data[i + nid_offset].gain;
//            sp_data[i].fea_missing_gh = missing_gh_data[i];
//            sp_data[i].default_right = 0;
            sp_data[i].nid = i + nid_offset;
            sp_data[i].no_split_value_update = true;
            sp_data[i].fea_missing_gh = nodes_data[nid_offset + i].sum_gh_pair;
            sp_data[i].rch_sum_gh = 0;
        } else {
            int_float bst = best_idx_gain_data[best_idx_gain_idx[i]];
            float_type best_split_gain = get < 1 > (bst);
            int split_index = get < 0 > (bst);
            if (!nodes_data[i + nid_offset].is_valid) {
                sp_data[i].split_fea_id = -1;
                sp_data[i].nid = -1;
                // todo: check, ThunderGBM uses return;
                continue;
            }

            int fid = hist_fid_data[split_index];
            int has_split_value = (n_bins[i + 1] != n_bins[i]);

            CHECK_EQ(fid, nodes_data[i + nid_offset].split_feature_id);
            sp_data[i].split_fea_id = nodes_data[i + nid_offset].split_feature_id;
            sp_data[i].nid = i + nid_offset;
            sp_data[i].gain = fabsf(best_split_gain);
//        int n_bins = cut.cut_points_val.size();
            int n_column = sorted_dataset.n_features();
            if (has_split_value) {
                sp_data[i].split_bid = (unsigned char) (split_index - n_bins[i]);
                sp_data[i].fval = cut_val_data[cut_col_ptr_data[fid] + split_index - n_bins[i]];
                sp_data[i].rch_sum_gh = hist_data[split_index];
            } else {
                sp_data[i].split_bid = nodes_data[i + nid_offset].split_bid;
                sp_data[i].fval = nodes_data[i + nid_offset].split_value;
                sp_data[i].rch_sum_gh = nodes_data[i + nid_offset].sum_gh_pair;
            }
//        sp_data[i].fval = cut_val_data[split_index % n_bins];
//        sp_data[i].split_bid = (unsigned char) (split_index % n_bins - cut_col_ptr_data[fid]);
            sp_data[i].fea_missing_gh = missing_gh_data[i];
            sp_data[i].default_right = best_split_gain < 0;
        }

    }
    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;
}

//todo: reduce hist size according to current level (not n_max_split)
void HistTreeBuilder::compute_histogram_in_a_level(int level, int n_max_splits, int n_bins, int n_nodes_in_level,
                                                   int *hist_fid, SyncArray<GHPair> &missing_gh,
                                                   SyncArray<GHPair> &hist) {
    std::chrono::high_resolution_clock timer;

    SyncArray<int> &nid = ins2node_id;
    SyncArray<GHPair> &gh_pair = gradients;
    Tree &tree = trees;
    SyncArray<SplitPoint> &sp = this->sp;
    HistCut &cut = this->cut;
    auto &dense_bin_id = this->dense_bin_id;
    auto &last_hist = this->last_hist;

    TIMED_FUNC(timerObj);
//    int n_nodes_in_level = static_cast<int>(pow(2, level));
    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = sorted_dataset.n_features();
    int n_partition = n_column * n_nodes_in_level;
//    int n_bins = cut.cut_points_val.size();
//    int n_max_nodes = 2 << param.depth;
//    int n_max_splits = n_max_nodes * n_bins;
    int n_split = n_nodes_in_level * n_bins;

    LOG(TRACE) << "start finding split";

    {
        TIMED_SCOPE(timerObj, "build hist");
        if (n_nodes_in_level == 1) {
            auto hist_data = hist.host_data();
            auto cut_col_ptr_data = cut.cut_col_ptr.host_data();
            auto gh_data = gh_pair.host_data();
            auto dense_bin_id_data = dense_bin_id.host_data();
            auto max_num_bin = param.max_num_bin;
            auto n_instances = this->n_instances;
//                ThunderGBM: check size of histogram.
            //has bug if using openmp
//            #pragma omp parallel for
            for (int i = 0; i < n_instances * n_column; i++) {
                int iid = i / n_column;
                int fid = i % n_column;
                unsigned char bid = dense_bin_id_data[iid * n_column + fid];
                if (bid != max_num_bin) {
                    int feature_offset = cut_col_ptr_data[fid];
                    const GHPair src = gh_data[iid];
                    GHPair &dest = hist_data[feature_offset + bid];
                    dest = dest + src;
//                    g and h values are 0 if after HE encryption
//                    if (src.h != 0) {
////                        #pragma omp atomic
//                        dest.h += src.h;
//                    }
//                    if (src.g != 0) {
////                        #pragma omp atomic
//                        dest.g += src.g;
//                    }

                }
            }
        } else {
            auto t_dp_begin = timer.now();
            SyncArray<int> node_idx(n_instances);
            SyncArray<int> node_ptr(n_nodes_in_level + 1);
            {
                TIMED_SCOPE(timerObj, "data partitioning");
                SyncArray<int> nid4sort(n_instances);
                nid4sort.copy_from(ins2node_id);
                sequence(thrust::host, node_idx.host_data(), node_idx.host_end(), 0);
                thrust::stable_sort_by_key(thrust::host, nid4sort.host_data(), nid4sort.host_end(),
                                           node_idx.host_data());
                auto counting_iter = thrust::make_counting_iterator<int>(nid_offset);
                node_ptr.host_data()[0] =
                        thrust::lower_bound(thrust::host, nid4sort.host_data(), nid4sort.host_end(), nid_offset) -
                        nid4sort.host_data();

                thrust::upper_bound(thrust::host, nid4sort.host_data(), nid4sort.host_end(), counting_iter,
                                    counting_iter + n_nodes_in_level, node_ptr.host_data() + 1);
            }
            auto t_dp_end = timer.now();
            std::chrono::duration<double> dp_used_time = t_dp_end - t_dp_begin;
            this->total_dp_time += dp_used_time.count();


            auto node_ptr_data = node_ptr.host_data();
            auto node_idx_data = node_idx.host_data();
            auto cut_col_ptr_data = cut.cut_col_ptr.host_data();
            auto gh_data = gh_pair.host_data();
            auto dense_bin_id_data = dense_bin_id.host_data();
            auto max_num_bin = param.max_num_bin;

            for (int i = 0; i < n_nodes_in_level / 2; ++i) {

                int nid0_to_compute = i * 2;
                int nid0_to_substract = i * 2 + 1;
                //node_ptr_data[i+1] - node_ptr_data[i] is the number of instances in node i, i is the node id in current level (start from 0)
                int n_ins_left = node_ptr_data[nid0_to_compute + 1] - node_ptr_data[nid0_to_compute];
                int n_ins_right = node_ptr_data[nid0_to_substract + 1] - node_ptr_data[nid0_to_substract];
                if (std::max(n_ins_left, n_ins_right) == 0) continue;
                //only compute the histogram on the node with the smaller data
                if (n_ins_left > n_ins_right)
                    std::swap(nid0_to_compute, nid0_to_substract);
                //compute histogram
                {
                    int nid0 = nid0_to_compute;
                    auto idx_begin = node_ptr.host_data()[nid0];
                    auto idx_end = node_ptr.host_data()[nid0 + 1];
                    auto hist_data = hist.host_data() + nid0 * n_bins;
                    this->total_hist_num++;
                    //                ThunderGBM: check size of histogram.
                    //has bug if using openmp
//#pragma omp parallel for
                    for (int i = 0; i < (idx_end - idx_begin) * n_column; i++) {

                        int iid = node_idx_data[i / n_column + idx_begin];
                        int fid = i % n_column;
                        unsigned char bid = dense_bin_id_data[iid * n_column + fid];
                        if (bid != max_num_bin) {
                            int feature_offset = cut_col_ptr_data[fid];
                            const GHPair src = gh_data[iid];
                            GHPair &dest = hist_data[feature_offset + bid];
//                            if (src.h != 0) {
////                                #pragma omp atomic
//                                dest.h += src.h;
//                            }
//                            if (src.g != 0) {
////                                #pragma omp atomic
//                                dest.g += src.g;
//                            }
                            dest = dest + src;
                        }
                    }
                }

                //subtract to the histogram of the other node
                auto t_copy_start = timer.now();
                {
                    auto hist_data_computed = hist.host_data() + nid0_to_compute * n_bins;
                    auto hist_data_to_compute = hist.host_data() + nid0_to_substract * n_bins;
                    auto father_hist_data = last_hist.host_data() + (nid0_to_substract / 2) * n_bins;
//#pragma omp parallel for
                    for (int i = 0; i < n_bins; i++) {
                        hist_data_to_compute[i] = father_hist_data[i] - hist_data_computed[i];
                    }
                }
                auto t_copy_end = timer.now();
                std::chrono::duration<double> cp_used_time = t_copy_end - t_copy_start;
                this->total_copy_time += cp_used_time.count();
//                            PERFORMANCE_CHECKPOINT(timerObj);
            }  // end for each node
        }
        last_hist.resize(n_nodes_in_level * n_bins);
        auto last_hist_data = last_hist.host_data();
        auto hist_data = hist.host_data();
        for (int i = 0; i < n_nodes_in_level * n_bins; i++) {
            last_hist_data[i] = hist_data[i];
        }
    }

    this->build_n_hist++;
    inclusive_scan_by_key(thrust::host, hist_fid, hist_fid + n_split,
                          hist.host_data(), hist.host_data());
    LOG(DEBUG) << hist;

    auto nodes_data = tree.nodes.host_data();
    auto missing_gh_data = missing_gh.host_data();
    auto cut_col_ptr = cut.cut_col_ptr.host_data();
    auto hist_data = hist.host_data();
//#pragma omp parallel for
    for (int pid = 0; pid < n_partition; pid++) {
        int nid0 = pid / n_column;
        int nid = nid0 + nid_offset;
        //            todo: check, ThunderGBM uses return;
        if (!nodes_data[nid].splittable()) continue;
        int fid = pid % n_column;
        if (cut_col_ptr[fid + 1] != cut_col_ptr[fid]) {
            GHPair node_gh = hist_data[nid0 * n_bins + cut_col_ptr[fid + 1] - 1];
            missing_gh_data[pid] = nodes_data[nid].sum_gh_pair - node_gh;
        }
    }
    return;
}


void
HistTreeBuilder::compute_gain_in_a_level(SyncArray<float_type> &gain, int n_nodes_in_level, int n_bins, int *hist_fid,
                                         SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist, int n_column) {
//    SyncArray<compute_gainfloat_type> gain(n_max_splits);
    if (n_column == 0)
        n_column = sorted_dataset.n_features();
    int n_split = n_nodes_in_level * n_bins;
    int nid_offset = static_cast<int>(n_nodes_in_level - 1);
    auto compute_gain = []__host__(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight,
                                   float_type lambda) -> float_type {
        if (lch.h >= min_child_weight && rch.h >= min_child_weight)
            return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda) -
                   (father.g * father.g) / (father.h + lambda);
        else
            return 0;
    };
    const Tree::TreeNode *nodes_data = trees.nodes.host_data();
    GHPair *gh_prefix_sum_data = hist.host_data();
    float_type *gain_data = gain.host_data();
    const auto missing_gh_data = missing_gh.host_data();
//    auto ignored_set_data = ignored_set.host_data();
    //for lambda expression
    float_type mcw = param.min_child_weight;
    float_type l = param.lambda;

//#pragma omp parallel for
    for (int i = 0; i < n_split; i++) {
        int nid0 = i / n_bins;
        int nid = nid0 + nid_offset;
        int fid = hist_fid[i % n_bins];
        if (nodes_data[nid].is_valid) {
            int pid = nid0 * n_column + fid;
            GHPair father_gh = nodes_data[nid].sum_gh_pair;
            GHPair p_missing_gh = missing_gh_data[pid];
            GHPair rch_gh = gh_prefix_sum_data[i];
            float_type default_to_left_gain = std::max(0.f,
                                                       compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
//            rch_gh = rch_gh + p_missing_gh;
            rch_gh.g += p_missing_gh.g;
            rch_gh.h += p_missing_gh.h;
            float_type default_to_right_gain = std::max(0.f,
                                                        compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
            if (default_to_left_gain > default_to_right_gain)
                gain_data[i] = default_to_left_gain;
            else
                gain_data[i] = -default_to_right_gain;//negative means default split to right
        } else gain_data[i] = 0;
    }
    return;
}

void HistTreeBuilder::get_best_gain_in_a_level(SyncArray<float_type> &gain, SyncArray<int_float> &best_idx_gain,
                                               int n_nodes_in_level, int n_bins) {
    using namespace thrust;
    int n_split = n_nodes_in_level * n_bins;
    {
        TIMED_SCOPE(timerObj, "get best gain");
        auto arg_abs_max = []__host__(const int_float &a, const int_float &b) {
            if (fabsf(thrust::get<1>(a)) == fabsf(thrust::get<1>(b)))
                return thrust::get<0>(a) < thrust::get<0>(b) ? a : b;
            else
                return fabsf(thrust::get<1>(a)) > fabsf(thrust::get<1>(b)) ? a : b;
        };

        auto nid_iterator = thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                            thrust::placeholders::_1 / n_bins);

        reduce_by_key(
                thrust::host,
                nid_iterator, nid_iterator + n_split,
                make_zip_iterator(make_tuple(thrust::counting_iterator<int>(0), gain.host_data())),
                make_discard_iterator(),
                best_idx_gain.host_data(),
                thrust::equal_to<int>(),
                arg_abs_max
        );
        LOG(DEBUG) << n_split;
        LOG(DEBUG) << "best rank & gain = " << best_idx_gain;
    }
    return;
}


void HistTreeBuilder::get_split_points(SyncArray<int_float> &best_idx_gain, int n_nodes_in_level, int *hist_fid,
                                       SyncArray<GHPair> &missing_gh, SyncArray<GHPair> &hist) {
//    TIMED_SCOPE(timerObj, "get split points");
    int nid_offset = static_cast<int>(n_nodes_in_level - 1);
    const int_float *best_idx_gain_data = best_idx_gain.host_data();
    auto hist_data = hist.host_data();
    const auto missing_gh_data = missing_gh.host_data();
    auto cut_val_data = cut.cut_points_val.host_data();

    sp.resize(n_nodes_in_level);
    auto sp_data = sp.host_data();
    auto nodes_data = trees.nodes.host_data();

    auto cut_col_ptr_data = cut.cut_col_ptr.host_data();
#pragma omp parallel for
    for (int i = 0; i < n_nodes_in_level; i++) {
        int_float bst = best_idx_gain_data[i];
        float_type best_split_gain = get < 1 > (bst);
        int split_index = get < 0 > (bst);
        if (!nodes_data[i + nid_offset].is_valid) {
            sp_data[i].split_fea_id = -1;
            sp_data[i].nid = -1;
            // todo: check, ThunderGBM uses return;
            continue;
        }
        int fid = hist_fid[split_index];
        sp_data[i].split_fea_id = fid;
        sp_data[i].nid = i + nid_offset;
        sp_data[i].gain = fabsf(best_split_gain);
        int n_bins = cut.cut_points_val.size();
        int n_column = sorted_dataset.n_features();
        sp_data[i].fval = cut_val_data[split_index % n_bins];
        sp_data[i].split_bid = (unsigned char) (split_index % n_bins - cut_col_ptr_data[fid]);
        sp_data[i].fea_missing_gh = missing_gh_data[i * n_column + hist_fid[split_index]];
        sp_data[i].default_right = best_split_gain < 0;
        sp_data[i].rch_sum_gh = hist_data[split_index];
        sp_data[i].no_split_value_update = 0;
    }
    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;
}

void HistTreeBuilder::get_split_points_in_a_node(int node_id, int best_idx, float best_gain, int n_nodes_in_level,
                                                 int *hist_fid, SyncArray<GHPair> &missing_gh,
                                                 SyncArray<GHPair> &hist) {
//    TIMED_SCOPE(timerObj, "get split points");
    auto hist_data = hist.host_data();
    auto missing_gh_data = missing_gh.host_data();
    auto cut_val_data = cut.cut_points_val.host_data();

//    sp.resize(n_nodes_in_level);
    auto sp_data = sp.host_data();
    auto nodes_data = trees.nodes.host_data();

    auto cut_col_ptr_data = cut.cut_col_ptr.host_data();

    if (!nodes_data[node_id].is_valid) {
        sp_data[node_id].split_fea_id = -1;
        sp_data[node_id].nid = -1;
        return;
    }
    int fid = hist_fid[best_idx];
    sp_data[node_id].split_fea_id = fid;
    sp_data[node_id].nid = node_id + n_nodes_in_level - 1;
    sp_data[node_id].gain = fabsf(best_gain);
    int n_bins = cut.cut_points_val.size();
    int n_column = sorted_dataset.n_features();
    sp_data[node_id].fval = cut_val_data[best_idx % n_bins];
    sp_data[node_id].split_bid = (unsigned char) (best_idx % n_bins - cut_col_ptr_data[fid]);
    sp_data[node_id].fea_missing_gh = missing_gh_data[node_id * n_column + hist_fid[best_idx]];
    sp_data[node_id].default_right = best_gain < 0;
    sp_data[node_id].rch_sum_gh = hist_data[best_idx];
    sp_data[node_id].no_split_value_update = 0;

//    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;
}

void HistTreeBuilder::update_ins2node_id() {
    TIMED_FUNC(timerObj);
    SyncArray<bool> has_splittable(1);
//    auto &columns = shards.columns;
    //set new node id for each instance
    {
//        TIMED_SCOPE(timerObj, "get new node id");
        auto nid_data = ins2node_id.host_data();
        Tree::TreeNode *nodes_data = trees.nodes.host_data();
        has_splittable.host_data()[0] = false;
        bool *h_s_data = has_splittable.host_data();
        int column_offset = 0;

        int n_column = sorted_dataset.n_features();
        auto dense_bin_id_data = dense_bin_id.host_data();
        int max_num_bin = param.max_num_bin;
//#pragma omp parallel for
        for (int iid = 0; iid < n_instances; iid++) {
            int nid = nid_data[iid];
            const Tree::TreeNode &node = nodes_data[nid];
            int split_fid = node.split_feature_id;
            if (node.splittable() && ((split_fid - column_offset < n_column) && (split_fid >= column_offset))) {
                h_s_data[0] = true;
                unsigned char split_bid = node.split_bid;
                unsigned char bid = dense_bin_id_data[iid * n_column + split_fid - column_offset];
                bool to_left = true;
                if ((bid == max_num_bin && node.default_right) || (bid <= split_bid))
                    to_left = false;
                if (to_left) {
                    //goes to left child
                    nid_data[iid] = node.lch_index;
//                    #pragma omp atomic
                    nodes_data[node.lch_index].n_instances += 1;
                } else {
                    //right child
                    nid_data[iid] = node.rch_index;
//                    #pragma omp atomic
                    nodes_data[node.rch_index].n_instances += 1;
                }
            }
        }
    }
    LOG(DEBUG) << "new tree_id = " << ins2node_id;
    has_split = has_splittable.host_data()[0];
}

bool HistTreeBuilder::update_ins2node_id_in_a_node(int node_id) {
    TIMED_FUNC(timerObj);
    SyncArray<bool> has_splittable(1);
//    auto &columns = shards.columns;
    //set new node id for each instance
    {
//        TIMED_SCOPE(timerObj, "get new node id");
        auto nid_data = ins2node_id.host_data();
        const Tree::TreeNode *nodes_data = trees.nodes.host_data();
        has_splittable.host_data()[0] = false;
        bool *h_s_data = has_splittable.host_data();
        int column_offset = 0;

        int n_column = sorted_dataset.n_features();
        auto dense_bin_id_data = dense_bin_id.host_data();
        int max_num_bin = param.max_num_bin;
        vector<int> instances = {};
        const Tree::TreeNode &node = nodes_data[node_id];
        for (int iid = 0; iid < n_instances; iid++)
            if (nid_data[iid] == node_id)
                instances.push_back(iid);
#pragma omp parallel for
        for (int idx = 0; idx < instances.size(); idx++) {
            int iid = instances[idx];
            int split_fid = node.split_feature_id;
            if (node.splittable() && ((split_fid - column_offset < n_column) && (split_fid >= column_offset))) {
                h_s_data[0] = true;
                unsigned char split_bid = node.split_bid;
                unsigned char bid = dense_bin_id_data[iid * n_column + split_fid - column_offset];
                bool to_left = true;
                if ((bid == max_num_bin && node.default_right) || (bid <= split_bid))
                    to_left = false;
                if (to_left) {
                    //goes to left child
                    nid_data[iid] = node.lch_index;
                } else {
                    //right child
                    nid_data[iid] = node.rch_index;
                }
            }
        }
    }
//    LOG(DEBUG) << "new tree_id = " << ins2node_id;
    return has_splittable.host_data()[0];
}

//for each node
void HistTreeBuilder::compute_histogram_in_a_node(SyncArray<GHPair> &gradients, HistCut &cut,
                                                  SyncArray<unsigned char> &dense_bin_id) {
    int n_columns = cut.cut_col_ptr.size() - 1;
    int n_instances = dense_bin_id.size() / n_columns;
    auto gh_data = gradients.host_data();
    auto cut_col_ptr_data = cut.cut_col_ptr.host_data();
    auto dense_bin_id_data = dense_bin_id.host_data();
    int n_bins = n_columns + cut_col_ptr_data[n_columns];

    SyncArray<GHPair> hist(n_bins);
    auto hist_data = hist.host_data();

    for (int i = 0; i < n_instances * n_columns; i++) {
        int iid = i / n_columns;
        int fid = i % n_columns;
        unsigned char bid = dense_bin_id_data[iid * n_columns + fid];

        int feature_offset = cut_col_ptr_data[fid] + fid;
        const GHPair src = gh_data[iid];
        GHPair &dest = hist_data[feature_offset + bid];
        dest = dest + src;
    }

    last_hist.resize(n_bins);
    last_hist.copy_from(hist);
}


//assumption: GHPairs in the histograms of all clients are arranged in the same order

void HistTreeBuilder::merge_histograms_server_propose(SyncArray<GHPair> &merged_hist, SyncArray<GHPair> &merged_missing_gh) {
    int n_bins = parties_hist[0].size();
    CHECK_EQ(parties_hist[0].size(), parties_hist[1].size());
    int n_size = parties_missing_gh[0].size();
    merged_hist.resize(n_bins);
    merged_missing_gh.resize(n_size);
//    SyncArray<GHPair> merged_hist(n_bins);
//    SyncArray<GHPair> merged_missing_gh(n_size);
    auto merged_hist_data = merged_hist.host_data();
    auto merged_missing_gh_data = merged_missing_gh.host_data();

    for (int i = 0; i < parties_hist.size(); i++) {
        auto hist_data = parties_hist[i].host_data();
        int n_bins = parties_hist[i].size();
        //thrust::transform(merged_hist_data, merged_hist_data + n_bins,
        //                  hist_data, merged_hist_data, thrust::plus<GHPair>());
#pragma omp parallel for
        for (int j = 0; j < n_bins; j++) {
            GHPair &src = hist_data[j];
            GHPair &hist_dest = merged_hist_data[j];
            hist_dest = hist_dest + src;
        }
    }

    for (int i = 0; i < parties_missing_gh.size(); i++) {
        auto missing_gh_data =  parties_missing_gh[i].host_data();
        //thrust::transform(merged_missing_gh_data, merged_missing_gh_data + n_size,
        //        missing_gh_data, merged_missing_gh_data, thrust::plus<GHPair>());
#pragma omp parallel for
        for (int j = 0; j < n_size; j++) {
            GHPair &missing_gh = missing_gh_data[j];
            GHPair &missing_gh_dest = merged_missing_gh_data[j];
            missing_gh_dest = missing_gh_dest + missing_gh;
        }
    }

//    hist.resize(n_bins);
//    hist.copy_from(merged_hist);
//   // LOG(INFO) << "MERGE HIST: " << last_hist;
//    missing_gh.resize(n_size);
//    missing_gh.copy_from(merged_missing_gh);
//    last_hist.resize(n_bins);
//    last_hist.copy_from(merged_hist);
}


void HistTreeBuilder::merge_histograms_client_propose(SyncArray<GHPair> &hist, SyncArray<GHPair> &missing_gh, vector<vector<vector<float>>> feature_range, int n_max_splits) {

    float inf = std::numeric_limits<float>::infinity();
    // find feature range of each feature for each party
    int n_columns = parties_cut[0].cut_col_ptr.size() - 1;
    vector<vector<float>> ranges(n_columns);

    // Merging all cut points into one single cut points
    for (int n = 0; n < n_columns; n++) {
        for (int p = 0; p < parties_hist.size(); p++) {
            auto cut_col_data = parties_cut[p].cut_col_ptr.host_data();
            auto cut_points_val_data = parties_cut[p].cut_points_val.host_data();

            int column_start = cut_col_data[n];
            int column_end = cut_col_data[n+1];

            for (int i = column_start; i < column_end; i++) {
                ranges[n].push_back(cut_points_val_data[i]);
            }
        }
    }

    // Once we have gathered the sorted range, we can randomly sample the cut points to match with the number of bins
    SyncArray<float> cut_points_val;
    SyncArray<int> cut_col_ptr;
    int n_features = ranges.size();
    int max_num_bins = parties_cut[0].cut_points_val.size() / n_columns + 1;
    cut_points_val.resize(n_features * max_num_bins);
    cut_col_ptr.resize(n_features + 1);

    auto cut_points_val_data = cut_points_val.host_data();
    auto cut_col_ptr_data = cut_col_ptr.host_data();

    int index = 0;

    for (int fid = 0; fid < n_features; fid++) {
        vector<float> sample;
        cut_col_ptr_data[fid] = index;

        // Always keep the maximum value
        auto max_element = *std::max_element(ranges[fid].begin(), ranges[fid].end());
        sample.push_back(max_element);

        // Randomly sample number of cut point according to max num bins
        unsigned seed = 0;
        std::shuffle(ranges[fid].begin(), ranges[fid].end(), std::default_random_engine(seed));

        struct compare
        {
            int key;
            compare(int const &i): key(i) {}

            bool operator()(int const &i) {
                return (i == key);
            }
        };


        for (int i = 0; i < ranges[fid].size(); i++) {

            if (sample.size() == max_num_bins)
                break;

            auto element = ranges[fid][i];
            // Check if element already in cut points val data
            if (not (std::find(sample.begin(), sample.end(), element) != sample.end()))
                sample.push_back(element);
        }

        // Sort the sample in descending order
        std::sort(sample.begin(), sample.end(), std::greater<float>());

        // Populate cut points val with samples
        for (int i = 0; i < sample.size(); i++) {
            cut_points_val_data[index] = sample[i];
            index++;
        }
    }
    cut_col_ptr_data[n_features] = index;

    SyncArray<GHPair> merged_hist(n_max_splits);
    auto merged_hist_data = merged_hist.host_data();
    int n_max_nodes = n_max_splits / (n_columns * max_num_bins);


    // Populate histogram based on generated cut points
    for (int node_offset = 0; node_offset < n_max_nodes; node_offset++) {
        // For each feature
        for (int fid = 0; fid < n_columns; fid++) {
            // Get global columns and values of feature
            auto cut_col_ptr_data = cut_col_ptr.host_data();
            auto cut_points_val_data = cut_points_val.host_data();
            int column_start = cut_col_ptr_data[fid];
            int column_end = cut_col_ptr_data[fid + 1];

            // Get range of global cut point of the feature
            SyncArray<float> cut_points_range(column_end - column_start);
            auto cut_points_range_data = cut_points_range.host_data();
            for (int p = column_start; p < column_end; p++) {
                cut_points_range_data[p - column_start] = cut_points_val_data[p];
            }

            // Get minimum value of feature
            auto global_feature_min_value = *std::min_element(ranges[fid].begin(), ranges[fid].end());

            // For each party histogram
            for (int pid = 0; pid < parties_hist.size(); pid++) {
                // Get corresponding column and value array
                auto parties_hist_data = parties_hist[pid].host_data();
                auto parties_cut_col_ptr_data = parties_cut[pid].cut_col_ptr.host_data();
                auto parties_cut_points_val_data = parties_cut[pid].cut_points_val.host_data();

                int party_column_start = parties_cut_col_ptr_data[fid];
                int party_column_end = parties_cut_col_ptr_data[fid + 1];
                SyncArray<float> party_cut_points_range(party_column_end - party_column_start);
                auto party_cut_points_range_data = party_cut_points_range.host_data();
                for (int p = party_column_start; p < party_column_end; p++) {
                    party_cut_points_range_data[p - party_column_start] = parties_cut_points_val_data[p];
                }

                // For each global range of current feature
                for (int index = 0; index < cut_points_range.size(); index++) {
                        float_type upper_bound = cut_points_range_data[index];
                        float_type lower_bound;
                        if (index == cut_points_range.size() - 1) {
                            lower_bound = global_feature_min_value;
                        }else {
                            lower_bound = cut_points_range_data[index + 1];
                        }


                        // for each range pair of current feature
                        for (int i = 0; i < party_cut_points_range.size(); i++) {

                                float_type client_high = party_cut_points_range_data[i];
                                float_type client_low;
                                if (i == party_cut_points_range.size() - 1) {
                                     client_low = feature_range[fid][pid][0];
                                }else {
                                     client_low = party_cut_points_range_data[i + 1];
                                }

                                int node_offset_index = node_offset * (max_num_bins * n_columns);
                                int dest_index = node_offset_index + column_start + index;
                                int src_index = node_offset_index + party_column_start + i;

                                // Case 1
                                if (client_low >= lower_bound && upper_bound <= client_high) {
                                    GHPair &dest = merged_hist_data[dest_index];
                                    GHPair &src = parties_hist_data[src_index];
                                    dest.g += src.g;
                                    dest.h += src.h;
                                // Case 3
                                } else if (client_low < lower_bound && upper_bound <= client_high) {
                                    float_type factor = (client_high - lower_bound) / (client_high - client_low);
                                    GHPair &dest = merged_hist_data[dest_index];
                                    GHPair &src = parties_hist_data[src_index];
                                    dest.g += src.g * factor;
                                    dest.h += src.h * factor;
                                // Case 2
                                } else if (client_high > upper_bound && lower_bound <= client_low) {
                                    float_type factor = (upper_bound - client_low) / (client_high - client_low);
                                    GHPair &dest = merged_hist_data[dest_index];
                                    GHPair &src = parties_hist_data[src_index];
                                    dest.g += src.g * factor;
                                    dest.h += src.h * factor;
                                // Case 4
                                } else if (client_low < lower_bound && client_high > upper_bound) {
                                    float_type factor = (upper_bound - lower_bound) / (client_high - client_low);
                                    GHPair &dest = merged_hist_data[dest_index];
                                    GHPair &src = parties_hist_data[src_index];
                                    dest.g += src.g * factor;
                                    dest.h += src.h * factor;
                                }
                        }
                    }
                }
            }
        }

    // Merge missing gh by summing
    int n_size = parties_missing_gh[0].size();
    SyncArray<GHPair> merged_missing_gh(n_size);
    auto merged_missing_gh_data = merged_missing_gh.host_data();

#pragma omp parallel for
    for (int i = 0; i < parties_missing_gh.size(); i++) {
        auto missing_gh_data =  parties_missing_gh[i].host_data();
#pragma omp parallel for
        for (int j = 0; j < n_size; j++) {
            GHPair &missing_gh = missing_gh_data[j];
            GHPair &missing_gh_dest = merged_missing_gh_data[j];
            missing_gh_dest = missing_gh_dest + missing_gh;
        }
    }
    hist.resize(merged_hist.size());
    hist.copy_from(merged_hist);
    missing_gh.resize(n_size);
    missing_gh.copy_from(merged_missing_gh);
}

//assumption 1: bin sizes for the split of a feature are the same
//assumption 2: for each feature, there must be at least 3 bins (2 cut points)
//assumption 3: cut_val_data is sorted by feature id and split value, eg: [f0(0.1), f0(0.2), f0(0.3), f1(100), f1(200),...]
//assumption 4: gradients and hessians are near uniformly distributed


//void HistTreeBuilder::merge_histograms_client_propose(SyncArray<GHPair> &hist, SyncArray<GHPair> &missing_gh, int n_max_splits) {
//    CHECK_EQ(parties_hist.size(), parties_cut.size());
//    int n_columns = parties_cut[0].cut_col_ptr.size() - 1;
//    vector<float_type> low(n_columns, std::numeric_limits<float>::max());
//    vector<float_type> high(n_columns, -std::numeric_limits<float>::max());
//    vector<float_type> resolution(n_columns, std::numeric_limits<float>::max());
//    vector<vector<float_type>> bin_edges;
//
//    for (int i = 0; i < parties_cut.size(); i++) {
//        auto cut_val_data = parties_cut[i].cut_points_val.host_data();
//        auto cut_col_ptr_data = parties_cut[i].cut_col_ptr.host_data();
//        vector<float_type> v = {};
//        for (int j = 0; j < n_columns; j++) {
//            int end = cut_col_ptr_data[j + 1];
//            int start = cut_col_ptr_data[j];
//            float_type res = cut_val_data[end - 1] - cut_val_data[end - 2];
//            resolution[j] = std::min(res, resolution[j]);
//            float_type l = cut_val_data[start] - res;
//            low[j] = std::min(l, low[j]);
//            float_type h = cut_val_data[end - 1] + res;
//            high[j] = std::max(h, high[j]);
//            for (int k = -1; k < end - start + 1; k++)
//                v.push_back(cut_val_data[start] + k * res);
//        }
//        bin_edges.push_back(v);
//    }
//
//    int n_bins = 0;
//    vector<float_type> merged_bin_edges;
//    vector<int> merged_bins_count;
//    merged_bins_count.push_back(0);
//    for (int i = 0; i < n_columns; i++) {
//        float_type count = (high[i] - low[i]) / resolution[i];
//        if (abs(int(count) - count) < 1e-6)
//            count = int(count);
//        else
//            count = ceil(count);
//        n_bins += count;
//        merged_bins_count.push_back(n_bins);
//        for (int j = 0; j <= count; j++)
//            merged_bin_edges.push_back(std::min(low[i] + j * resolution[i], high[i]));
//    }
//    LOG(INFO) << merged_bin_edges;
//    LOG(INFO) << n_bins;
//    SyncArray<GHPair> merged_hist(n_bins);
//    auto merged_hist_data = merged_hist.host_data();
//
//    LOG(INFO) << parties_hist[0].size() / n_max_splits;
//    LOG(INFO) << parties_cut[0].cut_col_ptr.size();
//    LOG(INFO) << parties_cut[0].cut_points_val.size();
//    LOG(INFO) << bin_edges[0].size();
//
//    for (int i = 0; i < parties_hist.size(); i++) {
//        CHECK_EQ(parties_hist[i].size() / n_max_splits, parties_cut[i].cut_points_val.size());
//        CHECK_EQ(parties_hist[i].size() / n_max_splits + n_columns, bin_edges[i].size());
//        auto hist_data = parties_hist[i].host_data();
//        auto cut_col_ptr_data = parties_cut[i].cut_col_ptr.host_data();
//        for (int j = 0; j < n_columns; j++) {
//            int client_bin_index_low = cut_col_ptr_data[j] + 2 * j;
//            int client_bin_index_high = cut_col_ptr_data[j + 1] + 2 * (j + 1);
//            for (int k = merged_bins_count[j]; k < merged_bins_count[j + 1]; k++) {
//                float_type bin_low = merged_bin_edges[k + j];
//                float_type bin_high = merged_bin_edges[k + j + 1];
//                for (int m = client_bin_index_low; m < client_bin_index_high - 1; m++) {
//                    float_type client_low = bin_edges[i][m];
//                    float_type client_high = bin_edges[i][m + 1];
//                    if (bin_low < client_low && bin_high > client_low) {
//                        GHPair &dest = merged_hist_data[k];
//                        GHPair &src = hist_data[m - j];
//                        float_type factor = (bin_high - client_low) / (client_high - client_low);
//                        dest.g += src.g * factor;
//                        dest.h += src.h * factor;
//                    } else if (bin_low >= client_low && bin_high <= client_high) {
//                        GHPair &dest = merged_hist_data[k];
//                        GHPair &src = hist_data[m];
//                        float_type factor = (bin_high - bin_low) / (client_high - client_low);
//                        dest.g += src.g * factor;
//                        dest.h += src.h * factor;
//                    } else if (bin_high > client_high && bin_low < client_high) {
//                        GHPair &dest = merged_hist_data[k];
//                        GHPair &src = hist_data[m];
//                        float_type factor = (client_high - bin_low) / (client_high - client_low);
//                        dest.g += src.g * factor;
//                        dest.h += src.h * factor;
//                    }
//                }
//            }
//        }
//    }
////    hist.resize(n_bins);
////    hist.copy_from(merged_hist);
////    // LOG(INFO) << "MERGE HIST: " << last_hist;
////    missing_gh.resize(n_size);
//   // missing_gh.copy_from(merged_missing_gh);
////    hist.resize(n_bins);
////    hist.copy_from(merged_hist);
//}

void HistTreeBuilder::concat_histograms() {
    int n_bins = 0;
    vector<int> ptr = {0};
    for (int i = 0; i < parties_hist.size(); i++) {
        n_bins += parties_hist[i].size();
        ptr.push_back(ptr.back() + n_bins);
    }
    SyncArray<GHPair> concat_hist(n_bins);
    auto concat_hist_data = concat_hist.host_data();

    for (int i = 0; i < parties_hist.size(); i++) {
        auto hist_data = parties_hist[i].host_data();
        for (int j = 0; j < parties_hist[i].size(); j++) {
            concat_hist_data[ptr[i] + j] = hist_data[j];
        }
    }

    last_hist.resize(n_bins);
    last_hist.copy_from(concat_hist);
}

SyncArray<float_type> HistTreeBuilder::gain(Tree &tree, SyncArray<GHPair> &hist, int level, int n_split) {
    SyncArray<float_type> gain(n_split);
    const Tree::TreeNode *nodes_data = tree.nodes.host_data();
    float_type mcw = this->param.min_child_weight;
    float_type l = this->param.lambda;
    int nid_offset = static_cast<int>(pow(2, level) - 1);
//    SyncArray<GHPair> missing_gh(n_partition);
//    const auto missing_gh_data = missing_gh.host_data();
    GHPair *gh_prefix_sum_data = hist.host_data();
    float_type *gain_data = gain.host_data();
    auto compute_gain = []__host__(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight,
                                   float_type lambda) -> float_type {
        if (lch.h >= min_child_weight && rch.h >= min_child_weight)
            return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda) -
                   (father.g * father.g) / (father.h + lambda);
        else
            return 0;
    };
    for (int i = 0; i < n_split; i++) {
        int n_bins = hist.size();
        int nid = i / n_bins + nid_offset;
        if (nodes_data[nid].is_valid) {
            GHPair father_gh = nodes_data[nid].sum_gh_pair;
//            GHPair p_missing_gh = missing_gh_data[pid];
            GHPair rch_gh = gh_prefix_sum_data[i];
            float_type left_gain = std::max(0.f, compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
            gain_data[i] = left_gain;
//          rch_gh = rch_gh + p_missing_gh;
//          float_type default_to_right_gain = std::max(0.f,
//                                                   compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
//          if (default_to_left_gain > default_to_right_gain) {
//              gain_data[i] = default_to_left_gain;
//          } else {
//              gain_data[i] = -default_to_right_gain;//negative means default split to right
//          }
        } else {
            gain_data[i] = 0;
        }
    }
    return gain;
}
