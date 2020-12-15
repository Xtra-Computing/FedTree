//
// Created by liqinbin on 12/11/20.
//
#include "FedTree/Tree/hist_tree_builder.h"

#include "FedTree/util/cub_wrapper.h"
#include "FedTree/util/device_lambda.cuh"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/discard_iterator.h"
#include "thrust/sequence.h"
#include "thrust/binary_search.h"
#include "FedTree/util/multi_device.h"


void HistTreeBuilder::find_split(int level) {
    TIMED_FUNC(timerObj);
    int n_nodes_in_level = static_cast<int>(pow(2, level));
//    int nid_offset = static_cast<int>(pow(2, level) - 1);
//    int n_column = dataset.n_features_;
//    int n_partition = n_column * n_nodes_in_level;
    int n_bins = cut.cut_points_val.size();
    int n_max_nodes = 2 << param.depth;
    int n_max_splits = n_max_nodes * n_bins;

    auto cut_fid_data = cut.cut_fid.host_data();
    auto i2fid = [=] __host__(int i) { return cut_fid_data[i % n_bins]; };
    auto hist_fid = make_transform_iterator(counting_iterator<int>(0), i2fid);

    int n_split = n_nodes_in_level * n_bins;

    LOG(TRACE) << "start finding split";

    auto t_build_start = timer.now();

//    SyncArray<GHPair> hist(n_max_splits);
    SyncArray<float_type> gain(n_max_splits);
    compute_histogram_in_a_level(level, n_max_splits, n_bins, n_nodes_in_level, hist_fid);
    compute_gain_in_a_level(gain, n_max_splits, n_bins, hist_fid);
    SyncArray<int_float> best_idx_gain(n_nodes_in_level);
    get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins);
    get_split_points(best_idx_gain, n_nodes_in_level, hist_fid);
}




void HistTreeBuilder::compute_histogram_in_a_level(int level, int n_max_splits, int n_bins, int n_nodes_in_level, transform_iterator& hist_fid) {
    std::chrono::high_resolution_clock timer;

    SyncArray<int> &nid = ins2node_id;
    SyncArray<GHPair> &gh_pair = gradients;
    Tree &tree = trees;
    SyncArray<SplitPoint> &sp = this->sp;
    HistCut &cut = this->cut;
    auto &dense_bin_id = this->dense_bin_id;
    auto &last_hist = this->last_hist[device_id];

    TIMED_FUNC(timerObj);
//    int n_nodes_in_level = static_cast<int>(pow(2, level));
    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = dataset.n_features_;
    int n_partition = n_column * n_nodes_in_level;
//    int n_bins = cut.cut_points_val.size();
//    int n_max_nodes = 2 << param.depth;
//    int n_max_splits = n_max_nodes * n_bins;
    int n_split = n_nodes_in_level * n_bins;

    LOG(TRACE) << "start finding split";

    SyncArray<GHPair> hist(n_max_splits);
    SyncArray<GHPair> missing_gh(n_partition);

    {
        TIMED_SCOPE(timerObj, "build hist");
        if (n_nodes_in_level == 1){
            auto hist_data = hist.host_data();
            auto cut_row_ptr_data = cut.cut_row_ptr.host_data();
            auto gh_data = gh_pair.host_data();
            auto dense_bin_id_data = dense_bin_id.host_data();
            auto max_num_bin = param.max_num_bin;
            auto n_instances = this->n_instances;
//                ThunderGBM: check size of histogram.
//                #pragma omp parallel for
            for(int i = 0; i < n_instances * n_column; i++){
                int iid = i / n_column;
                int fid = i % n_column;
                unsigned char bid = dense_bin_id_data[iid * n_column + fid];
                if (bid != max_num_bin) {
                    int feature_offset = cut_row_ptr_data[fid];
                    const GHPair src = gh_data[iid];
                    GHPair &dest = hist_data[feature_offset + bid];
                    if(src.h != 0)
                        dest.h += src.h;
//                        todo: atomic add on cpu
//                            atomicAdd(&dest.h, src.h);
                    if(src.g != 0)
                        dest.g += src.g;
//                            atomicAdd(&dest.g, src.g);

                }
            }
        }
        else{
            auto t_dp_begin = timer.now();
            SyncArray<int> node_idx(n_instances);
            SyncArray<int> node_ptr(n_nodes_in_level + 1);
            {
                TIMED_SCOPE(timerObj, "data partitioning");
                SyncArray<int> nid4sort(n_instances);
                nid4sort.copy_from(ins2node_id);
                sequence(thrust::host, node_idx.host_data(), node_idx.host_end(), 0);

                thrust:sort_by_key(thrust:host, nid4sort.host_data(), nid4sort.host_end(), node_idx.host_data());
                auto counting_iter = make_counting_iterator < int > (nid_offset);
                node_ptr.host_data()[0] =
                        lower_bound(thrust:host, nid4sort.host_data(), nid4sort.host_end(), nid_offset) -
                        nid4sort.host_data();

                upper_bound(thrust:host, nid4sort.host_data(), nid4sort.host_end(), counting_iter,
                            counting_iter + n_nodes_in_level, node_ptr.host_data() + 1);
                LOG(DEBUG) << "node ptr = " << node_ptr;
            }
            auto t_dp_end = timer.now();
            std::chrono::duration<double> dp_used_time = t_dp_end - t_dp_begin;
            this->total_dp_time += dp_used_time.count();


            auto node_ptr_data = node_ptr.host_data();
            auto node_idx_data = node_idx.host_data();
            auto cut_row_ptr_data = cut.cut_row_ptr.host_data();
            auto gh_data = gh_pair.host_data();
            auto dense_bin_id_data = dense_bin_id.host_data();
            auto max_num_bin = param.max_num_bin;
            for (int i = 0; i < n_nodes_in_level / 2; ++i) {

                int nid0_to_compute = i * 2;
                int nid0_to_substract = i * 2 + 1;
                //node_ptr_data[i+1] - node_ptr_data[i] is the number of instances in node i, i is the node id in current level (start from 0)
                int n_ins_left = node_ptr_data[nid0_to_compute + 1] - node_ptr_data[nid0_to_compute];
                int n_ins_right = node_ptr_data[nid0_to_substract + 1] - node_ptr_data[nid0_to_substract];
                if (max(n_ins_left, n_ins_right) == 0) continue;
                //only compute the histogram on the node with the smaller data
                if (n_ins_left > n_ins_right)
                    swap(nid0_to_compute, nid0_to_substract);

                //compute histogram
                {
                    int nid0 = nid0_to_compute;
                    auto idx_begin = node_ptr.host_data()[nid0];
                    auto idx_end = node_ptr.host_data()[nid0 + 1];
                    auto hist_data = hist.host_data() + nid0 * n_bins;
                    this->total_hist_num++;
                    //                ThunderGBM: check size of histogram.
//                        #pragma omp parallel for
                    for(int i = 0; i < (idx_end - idx_begin) * n_column; i++){
                        int iid = node_idx_data[i / n_column + idx_begin];
                        int fid = i % n_column;
                        unsigned char bid = dense_bin_id_data[iid * n_column + fid];
                        if (bid != max_num_bin) {
                            int feature_offset = cut_row_ptr_data[fid];
                            const GHPair src = gh_data[iid];
                            GHPair &dest = hist_data[feature_offset + bid];
                            if(src.h != 0)
                                dest.h += src.h
                            if(src.g != 0)
                                dest.g += src.g
                        }
                    }
                }

                //subtract to the histogram of the other node
                auto t_copy_start = timer.now();
                {
                    auto hist_data_computed = hist.host_data() + nid0_to_compute * n_bins;
                    auto hist_data_to_compute = hist.host_data() + nid0_to_substract * n_bins;
                    auto father_hist_data = last_hist.host_data() + (nid0_to_substract / 2) * n_bins;
                    #pragma omp parallel for
                    for(int i = 0; i < n_bins; i++){
                        hist_data_to_compute[i] = father_hist_data[i] - hist_data_computed[i];
                    }
                }
                auto t_copy_end = timer.now();
                std::chrono::duration<double> cp_used_time = t_copy_end - t_copy_start;
                this->total_copy_time += cp_used_time.count();
//                            PERFORMANCE_CHECKPOINT(timerObj);
            }  // end for each node
        }
        last_hist.copy_from(hist);
    }

    this->build_n_hist++;
    inclusive_scan_by_key(thrust::host, hist_fid, hist_fid + n_split,
                          hist.host_data(), hist.host_data());
    LOG(DEBUG) << hist;

    auto nodes_data = tree.nodes.host_data();
    auto missing_gh_data = missing_gh.host_data();
    auto cut_row_ptr = cut.cut_row_ptr.host_data();
    auto hist_data = hist.host_data();
#pragma omp parallel for
    for(int pid = 0; pid < n_partition; pid++){
        int nid0 = pid / n_column;
        int nid = nid0 + nid_offset;
        if (!nodes_data[nid].splittable()) return;
        int fid = pid % n_column;
        if (cut_row_ptr[fid + 1] != cut_row_ptr[fid]) {
            GHPair node_gh = hist_data[nid0 * n_bins + cut_row_ptr[fid + 1] - 1];
            missing_gh_data[pid] = nodes_data[nid].sum_gh_pair - node_gh;
        }
    }
    LOG(DEBUG) << missing_gh;
    return hist;
}


void HistTreeBuilder::compute_gain_in_a_level(SyncArrary<float_type> &gain, int n_max_splits, int n_bins, transform_iterator& hist_fid){
//    SyncArray<float_type> gain(n_max_splits);
    const Tree::TreeNode *nodes_data = trees.nodes.host_data();
    GHPair *gh_prefix_sum_data = last_hist.host_data();
    float_type *gain_data = gain.host_data();
    const auto missing_gh_data = missing_gh.host_data();
//    auto ignored_set_data = ignored_set.host_data();
    //for lambda expression
    float_type mcw = param.min_child_weight;
    float_type l = param.lambda;

#pragma omp parallel for
    for(int i = 0; i < n_split; i++){
        int nid0 = i / n_bins;
        int nid = nid0 + nid_offset;
        int fid = hist_fid[i % n_bins];
        if (nodes_data[nid].is_valid) {
            int pid = nid0 * n_column + hist_fid[i];
            GHPair father_gh = nodes_data[nid].sum_gh_pair;
            GHPair p_missing_gh = missing_gh_data[pid];
            GHPair rch_gh = gh_prefix_sum_data[i];
            float_type default_to_left_gain = max(0.f,
                                                  compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
            rch_gh = rch_gh + p_missing_gh;
            float_type default_to_right_gain = max(0.f,
                                                   compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
            if (default_to_left_gain > default_to_right_gain)
                gain_data[i] = default_to_left_gain;
            else
                gain_data[i] = -default_to_right_gain;//negative means default split to right

        } else gain_data[i] = 0;
    }
    return;
}

void HistTreeBuilder::get_best_gain_in_a_level(SyncArray<float_type> &gain, SyncArray<int_float> &best_idx_gain, int n_nodes_in_level, int n_bins){
    int n_split = n_nodes_in_level*n_bins;
    {
        TIMED_SCOPE(timerObj, "get best gain");
        auto arg_abs_max = [](const int_float &a, const int_float &b) {
            if (fabsf(get<1>(a)) == fabsf(get<1>(b)))
                return get<0>(a) < get<0>(b) ? a : b;
            else
                return fabsf(get<1>(a)) > fabsf(get<1>(b)) ? a : b;
        };

        auto nid_iterator = make_transform_iterator(counting_iterator<int>(0), placeholders::_1 / n_bins);

        reduce_by_key(
                thrust::host,
                nid_iterator, nid_iterator + n_split,
                make_zip_iterator(make_tuple(counting_iterator<int>(0), gain.host_data())),
                make_discard_iterator(),
                best_idx_gain.host_data(),
                thrust::equal_to<int>(),
                arg_abs_max
        );
        LOG(DEBUG) << n_split;
        LOG(DEBUG) << "best rank & gain = " << best_idx_gain;
    }
    return best_idx_gain;
}


void HistTreeBuilder::get_split_points(SyncArray<int_float> &best_idx_gain, int n_nodes_in_level, transform_iterator& hist_fid){
    TIMED_SCOPE(timerObj, "get split points");

    const int_float *best_idx_gain_data = best_idx_gain.host_data();
    auto hist_data = last_hist.host_data();
    const auto missing_gh_data = missing_gh.host_data();
    auto cut_val_data = cut.cut_points_val.host_data();

    sp.resize(n_nodes_in_level);
    auto sp_data = sp.host_data();
    auto nodes_data = trees.nodes.host_data();

    auto cut_row_ptr_data = cut.cut_row_ptr.host_data();
#pragma omp parallel for
    for(int i = 0; i < n_nodes_in_level; i++){
        int_float bst = best_idx_gain_data[i];
        float_type best_split_gain = get<1>(bst);
        int split_index = get<0>(bst);
        if (!nodes_data[i + nid_offset].is_valid) {
            sp_data[i].split_fea_id = -1;
            sp_data[i].nid = -1;
            return;
        }
        int fid = hist_fid[split_index];
        sp_data[i].split_fea_id = fid;
        sp_data[i].nid = i + nid_offset;
        sp_data[i].gain = fabsf(best_split_gain);
        sp_data[i].fval = cut_val_data[split_index % n_bins];
        sp_data[i].split_bid = (unsigned char) (split_index % n_bins - cut_row_ptr_data[fid]);
        sp_data[i].fea_missing_gh = missing_gh_data[i * n_column + hist_fid[split_index]];
        sp_data[i].default_right = best_split_gain < 0;
        sp_data[i].rch_sum_gh = hist_data[split_index];
    }
    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;
}

//SyncArray<int_float> TreeBuilder::best_idx_gain_junhui(SyncArray<float_type> &gain, int n_bins, int level, int n_split) {
//    int n_nodes_in_level = static_cast<int>(pow(2, level));
//    SyncArray<int_float> best_idx_gain(n_nodes_in_level);
//    auto nid = [this, n_bins](int index) {
//        return index/n_bins;
//    };
//    auto arg_abs_max = [](const int_float &a, const int_float &b) {
//        if (fabsf(thrust::get<1>(a)) == fabsf(thrust::get<1>(b)))
//            return thrust::get<0>(a) < thrust::get<0>(b) ? a : b;
//        else
//            return fabsf(thrust::get<1>(a)) > fabsf(thrust::get<1>(b)) ? a : b;
//    };
//    auto nid_iterator = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), nid);
//    reduce_by_key(
//            thrust::host,
//            nid_iterator, nid_iterator + n_split,
//            make_zip_iterator(make_tuple(thrust::counting_iterator<int>(0), gain.host_data())),
//            thrust::make_discard_iterator(),
//            best_idx_gain.host_data(),
//            thrust::equal_to<int>(),
//            arg_abs_max
//    );
//
//    return best_idx_gain;
//}


float_type
TreeBuilder::compute_gain(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight, float_type lambda) {
    if (lch.h >= min_child_weight && rch.h >= min_child_weight) {
        return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda) -
               (father.g * father.g) / (father.h + lambda);
    } else {
        return 0;
    }
}

//for each node
SyncArray<GHPair> HistTreeBuilder::compute_histogram_in_a_node(SyncArray<GHPair> &gradients, HistCut &cut,
                                   SyncArray<unsigned char> &dense_bin_id) {
    int n_columns = cut.cut_row_ptr.size() - 1;
//    what if the dataset is sparse
    int n_instances = dense_bin_id.size() / n_columns;
    auto gh_data = gradients.host_data();
    auto cut_row_ptr_data = cut.cut_row_ptr.host_data();
    auto dense_bin_id_data = dense_bin_id.host_data();
    int n_bins = n_columns + cut_row_ptr_data[n_columns];

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