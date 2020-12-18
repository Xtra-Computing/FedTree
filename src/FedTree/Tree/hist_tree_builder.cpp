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


using namespace thrust;
void HistTreeBuilder::init(DataSet &dataset, const GBDTParam &param) {
    TreeBuilder::init(dataset, param);
    //TODO refactor
    //init shards
//    shards = vector<Shard>(n_device);
//    vector<std::unique_ptr<SparseColumns>> v_columns(param.n_device);
//    for (int i = 0; i < param.n_device; ++i) {
//        v_columns[i].reset(&shards[i].columns);
//        shards[i].ignored_set = SyncArray<bool>(dataset.n_features());
//    }
//    SparseColumns columns;
//    if(dataset.use_cpu)
//        columns.csr2csc_cpu(dataset, v_columns);
//    else
//        columns.csr2csc_gpu(dataset, v_columns);
    if (!dataset.has_csc)
        dataset.csr_to_csc();
//    cut = vector<HistCut>(param.n_device);
//    dense_bin_id = SyncArray<unsigned char>();
//    last_hist = MSyncArray<GHPair>(param.n_device);

    cut.get_cut_points_fast(dataset, param.max_num_bin, n_instances);
    last_hist.resize((2 << param.depth) * cut.cut_points_val.size());
    get_bin_ids();
}

void HistTreeBuilder::get_bin_ids() {
//    SparseColumns &columns = shards[device_id].columns;
    HistCut &cut = this->cut;
    auto &dense_bin_id = this->dense_bin_id;
    using namespace thrust;
    int n_column = dataset->n_features();
    int nnz = dataset->csr_val.size();
    auto cut_row_ptr = cut.cut_row_ptr.host_data();
    auto cut_points_ptr = cut.cut_points_val.host_data();
    auto csc_val_data = &(dataset->csc_val[0]);
    auto csc_col_ptr_data = &(dataset->csc_col_ptr[0]);
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
        for(int cid = 0; cid < n_column; cid++){
            for(int i = csc_col_ptr_data[cid]; i < csc_col_ptr_data[cid + 1]; i++){
                auto search_begin = cut_points_ptr + cut_row_ptr[cid];
                auto search_end = cut_points_ptr + cut_row_ptr[cid + 1];
                auto val = csc_val_data[i];
                bin_id_data[i] = lowerBound(search_begin, search_end, val) - search_begin;
            }
        }
    }

    auto max_num_bin = param.max_num_bin;
    dense_bin_id.resize(n_instances * n_column);
    auto dense_bin_id_data = dense_bin_id.host_data();
    auto csc_row_idx_data = dataset->csc_row_idx.data();
    #pragma omp parallel for
    for(int i = 0; i < n_instances * n_column; i++){
        dense_bin_id_data[i] = max_num_bin;
    }
    #pragma omp parallel for
    for(int fid = 0; fid < n_column; fid++){
        for(int i = csc_col_ptr_data[fid]; i < csc_col_ptr_data[fid+1]; i++){
            int row = csc_row_idx_data[i];
            unsigned char bid = bin_id_data[i];
            dense_bin_id_data[row * n_column + fid] = bid;
        }
    }
}

void HistTreeBuilder::find_split(int level) {
    TIMED_FUNC(timerObj);
    std::chrono::high_resolution_clock timer;
    int n_nodes_in_level = static_cast<int>(pow(2, level));
//    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = dataset->n_features();
    int n_partition = n_column * n_nodes_in_level;
    int n_bins = cut.cut_points_val.size();
    int n_max_nodes = 2 << param.depth;
    int n_max_splits = n_max_nodes * n_bins;

    auto cut_fid_data = cut.cut_fid.host_data();

//    auto i2fid = [=] __host__(int i) { return cut_fid_data[i % n_bins]; };
//    auto hist_fid = make_transform_iterator(counting_iterator<int>(0), i2fid);

    SyncArray<int> hist_fid(cut.cut_fid.size());
    auto hist_fid_data = hist_fid.host_data();

#pragma omp parallel for
    for(int i = 0; i < hist_fid.size(); i++)
        hist_fid_data[i] = cut_fid_data[i % n_bins];


    int n_split = n_nodes_in_level * n_bins;
    SyncArray<GHPair> missing_gh(n_partition);
    LOG(TRACE) << "start finding split";

    auto t_build_start = timer.now();

//    SyncArray<GHPair> hist(n_max_splits);
    SyncArray<float_type> gain(n_max_splits);
    compute_histogram_in_a_level(level, n_max_splits, n_bins, n_nodes_in_level, hist_fid_data, missing_gh);
    compute_gain_in_a_level(gain, n_max_splits, n_bins, hist_fid_data, missing_gh);
    SyncArray<int_float> best_idx_gain(n_nodes_in_level);
    get_best_gain_in_a_level(gain, best_idx_gain, n_nodes_in_level, n_bins);
    get_split_points(best_idx_gain, n_nodes_in_level, hist_fid_data, missing_gh);
}

void HistTreeBuilder::compute_histogram_in_a_level(int level, int n_max_splits, int n_bins, int n_nodes_in_level,
                                                   int* hist_fid, SyncArray<GHPair> &missing_gh) {
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
    int n_column = dataset->n_features();
    int n_partition = n_column * n_nodes_in_level;
//    int n_bins = cut.cut_points_val.size();
//    int n_max_nodes = 2 << param.depth;
//    int n_max_splits = n_max_nodes * n_bins;
    int n_split = n_nodes_in_level * n_bins;

    LOG(TRACE) << "start finding split";

    SyncArray<GHPair> hist(n_max_splits);

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
            #pragma omp parallel for
            for(int i = 0; i < n_instances * n_column; i++){
                int iid = i / n_column;
                int fid = i % n_column;
                unsigned char bid = dense_bin_id_data[iid * n_column + fid];
                if (bid != max_num_bin) {
                    int feature_offset = cut_row_ptr_data[fid];
                    const GHPair src = gh_data[iid];
                    GHPair &dest = hist_data[feature_offset + bid];

                    if(src.h != 0) {
                        #pragma omp atomic
                        dest.h += src.h;
                    }
                    if(src.g != 0) {
                        #pragma omp atomic
                        dest.g += src.g;
                    }

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

                thrust:sort_by_key(thrust::host, nid4sort.host_data(), nid4sort.host_end(), node_idx.host_data());
                auto counting_iter = thrust::make_counting_iterator < int > (nid_offset);
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
                    #pragma omp parallel for
                    for(int i = 0; i < (idx_end - idx_begin) * n_column; i++){
                        int iid = node_idx_data[i / n_column + idx_begin];
                        int fid = i % n_column;
                        unsigned char bid = dense_bin_id_data[iid * n_column + fid];
                        if (bid != max_num_bin) {
                            int feature_offset = cut_row_ptr_data[fid];
                            const GHPair src = gh_data[iid];
                            GHPair &dest = hist_data[feature_offset + bid];
                            if(src.h != 0) {
                                #pragma omp atomic
                                dest.h += src.h;
                            }
                            if(src.g != 0) {
                                #pragma omp atomic
                                dest.g += src.g;
                            }
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
        //            todo: check, ThunderGBM uses return;
        if (!nodes_data[nid].splittable()) continue;
        int fid = pid % n_column;
        if (cut_row_ptr[fid + 1] != cut_row_ptr[fid]) {
            GHPair node_gh = hist_data[nid0 * n_bins + cut_row_ptr[fid + 1] - 1];
            missing_gh_data[pid] = nodes_data[nid].sum_gh_pair - node_gh;
        }
    }
    LOG(DEBUG) << missing_gh;
    return;
}


void HistTreeBuilder::compute_gain_in_a_level(SyncArray<float_type> &gain, int n_nodes_in_level, int n_bins, int* hist_fid,
                                              SyncArray<GHPair> &missing_gh){
//    SyncArray<float_type> gain(n_max_splits);
    int n_column = dataset->n_features();
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
            float_type default_to_left_gain = std::max(0.f,
                                                  compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
            rch_gh = rch_gh + p_missing_gh;
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

void HistTreeBuilder::get_best_gain_in_a_level(SyncArray<float_type> &gain, SyncArray<int_float> &best_idx_gain, int n_nodes_in_level, int n_bins){
    using namespace thrust;
    int n_split = n_nodes_in_level*n_bins;
    {
        TIMED_SCOPE(timerObj, "get best gain");
        auto arg_abs_max = []__host__(const int_float &a, const int_float &b) {
            if (fabsf(thrust::get<1>(a)) == fabsf(thrust::get<1>(b)))
                return thrust::get<0>(a) < thrust::get<0>(b) ? a : b;
            else
                return fabsf(thrust::get<1>(a)) > fabsf(thrust::get<1>(b)) ? a : b;
        };

        auto nid_iterator = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), thrust::placeholders::_1 / n_bins);

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
                                       SyncArray<GHPair> &missing_gh){
    TIMED_SCOPE(timerObj, "get split points");
    int nid_offset = static_cast<int>(n_nodes_in_level - 1);
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
            // todo: check, ThunderGBM uses return;
            continue;
        }
        int fid = hist_fid[split_index];
        sp_data[i].split_fea_id = fid;
        sp_data[i].nid = i + nid_offset;
        sp_data[i].gain = fabsf(best_split_gain);
        int n_bins = cut.cut_points_val.size();
        int n_column = dataset->n_features();
        sp_data[i].fval = cut_val_data[split_index % n_bins];
        sp_data[i].split_bid = (unsigned char) (split_index % n_bins - cut_row_ptr_data[fid]);
        sp_data[i].fea_missing_gh = missing_gh_data[i * n_column + hist_fid[split_index]];
        sp_data[i].default_right = best_split_gain < 0;
        sp_data[i].rch_sum_gh = hist_data[split_index];
    }
    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;
}

void HistTreeBuilder::update_ins2node_id() {
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

        int n_column = dataset->n_features();
        auto dense_bin_id_data = dense_bin_id.host_data();
        int max_num_bin = param.max_num_bin;
        #pragma omp parallel for
        for(int iid = 0; iid < n_instances; iid++){
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
                } else {
                    //right child
                    nid_data[iid] = node.rch_index;
                }
            }
        }
    }
    LOG(DEBUG) << "new tree_id = " << ins2node_id;
    has_split = has_splittable.host_data()[0];
}

//for each node
void HistTreeBuilder::compute_histogram_in_a_node(SyncArray<GHPair> &gradients, HistCut &cut,
                                        SyncArray<unsigned char> &dense_bin_id, bool enc) {
    int n_columns = cut.cut_row_ptr.size() - 1;
    int n_instances = dense_bin_id.size() / n_columns;
    auto gh_data = gradients.host_data();
    auto cut_row_ptr_data = cut.cut_row_ptr.host_data();
    auto dense_bin_id_data = dense_bin_id.host_data();
    int n_bins = n_columns + cut_row_ptr_data[n_columns];

    SyncArray<GHPair> hist(n_bins);
    auto hist_data = hist.host_data();
    if (enc) {
        AdditivelyHE::PaillierPublicKey pk = gh_data[0].pk;
        for (int i = 0; i < n_bins; i++) {
            hist_data[i].homo_encrypt(pk);
        }
    }

    for (int i = 0; i < n_instances * n_columns; i++) {
        int iid = i / n_columns;
        int fid = i % n_columns;
        unsigned char bid = dense_bin_id_data[iid * n_columns + fid];

        int feature_offset = cut_row_ptr_data[fid] + fid;
        const GHPair src = gh_data[iid];
        GHPair &dest = hist_data[feature_offset + bid];
        if (enc)
            dest = dest.homo_add(src);
        else
            dest = dest + src;
    }

    last_hist.resize(n_bins);
    last_hist.copy_from(hist);
}


//assumption: GHPairs in the histograms of all clients are arranged in the same order

void HistTreeBuilder::merge_histograms_server_propose(MSyncArray<GHPair> &histograms, bool enc) {

    int n_bins = histograms[0].size();
    SyncArray<GHPair> merged_hist(n_bins);
    auto merged_hist_data = merged_hist.host_data();
    if (enc) {
        AdditivelyHE::PaillierPublicKey pk = histograms[0].host_data()[0].pk;
        for (int i = 0; i < n_bins; i++) {
            merged_hist_data[i].homo_encrypt(pk);
        }
    }

    for (int i = 0; i < histograms.size(); i++) {
        auto hist_data = histograms[i].host_data();
        for (int j = 0; j < n_bins; j++) {
            GHPair &src = hist_data[j];
            GHPair &dest = merged_hist_data[j];
            if (enc)
                dest = dest.homo_add(src);
            else
                dest = dest + src;
        }
    }

    last_hist.resize(n_bins);
    last_hist.copy_from(merged_hist);
}


//assumption 1: bin sizes for the split of a feature are the same
//assumption 2: for each feature, there must be at least 3 bins (2 cut points)
//assumption 3: cut_val_data is sorted by feature id and split value, eg: [f0(0.1), f0(0.2), f0(0.3), f1(100), f1(200),...]
//assumption 4: gradients and hessians are near uniformly distributed

void HistTreeBuilder::merge_histograms_client_propose(MSyncArray<GHPair> &histograms, vector<HistCut> &cuts, bool enc) {
    CHECK_EQ(histograms.size(), cuts.size());
    int n_columns = cuts[0].cut_row_ptr.size() - 1;
    vector<float_type> low(n_columns, std::numeric_limits<float>::max());
    vector<float_type> high(n_columns, -std::numeric_limits<float>::max());
    vector<float_type> resolution(n_columns, std::numeric_limits<float>::max());
    vector<vector<float_type>> bin_edges;
    for (int i = 0; i < cuts.size(); i++) {
        auto cut_val_data = cuts[i].cut_points_val.host_data();
        auto cut_row_ptr_data = cuts[i].cut_row_ptr.host_data();
        vector<float_type> v = {};
        for (int j = 0; j < n_columns; j++) {
            int end = cut_row_ptr_data[j + 1];
            int start = cut_row_ptr_data[j];
            float_type res = cut_val_data[end - 1] - cut_val_data[end - 2];
            resolution[j] = std::min(res, resolution[j]);
            float_type l = cut_val_data[start] - res;
            low[j] = std::min(l, low[j]);
            float_type h = cut_val_data[end - 1] + res;
            high[j] = std::max(h, high[j]);
            for (int k = -1; k < end - start + 1; k++)
                v.push_back(cut_val_data[start] + k * res);
        }
        bin_edges.push_back(v);
    }

    int n_bins = 0;
    vector<float_type> merged_bin_edges;
    vector<int> merged_bins_count;
    merged_bins_count.push_back(0);
    for (int i = 0; i < n_columns; i++) {
        float_type count = (high[i] - low[i]) / resolution[i];
        if (abs(int(count) - count) < 1e-6)
            count = int(count);
        else
            count = ceil(count);
        n_bins += count;
        merged_bins_count.push_back(n_bins);
        for (int j = 0; j <= count; j++)
            merged_bin_edges.push_back(std::min(low[i] + j * resolution[i], high[i]));
    }

    SyncArray<GHPair> merged_hist(n_bins);
    auto merged_hist_data = merged_hist.host_data();
    for (int i = 0; i < histograms.size(); i++) {
        CHECK_EQ(histograms[i].size(), cuts[i].cut_points_val.size() + n_columns);
        CHECK_EQ(histograms[i].size() + n_columns, bin_edges[i].size());
        auto hist_data = histograms[i].host_data();
        auto cut_row_ptr_data = cuts[i].cut_row_ptr.host_data();
        for (int j = 0; j < n_columns; j++) {
            int client_bin_index_low = cut_row_ptr_data[j] + 2 * j;
            int client_bin_index_high = cut_row_ptr_data[j + 1] + 2 * (j + 1);
            for (int k = merged_bins_count[j]; k < merged_bins_count[j + 1]; k++) {
                float_type bin_low = merged_bin_edges[k + j];
                float_type bin_high = merged_bin_edges[k + j + 1];
                for (int m = client_bin_index_low; m < client_bin_index_high - 1; m++) {
                    float_type client_low = bin_edges[i][m];
                    float_type client_high = bin_edges[i][m + 1];
                    if (bin_low < client_low && bin_high > client_low) {
                        GHPair &dest = merged_hist_data[k];
                        GHPair &src = hist_data[m - j];
                        float_type factor = (bin_high - client_low) / (client_high - client_low);
                        dest.g += src.g * factor;
                        dest.h += src.h * factor;
                    } else if (bin_low >= client_low && bin_high <= client_high) {
                        GHPair &dest = merged_hist_data[k];
                        GHPair &src = hist_data[m];
                        float_type factor = (bin_high - bin_low) / (client_high - client_low);
                        dest.g += src.g * factor;
                        dest.h += src.h * factor;
                    } else if (bin_high > client_high && bin_low < client_high) {
                        GHPair &dest = merged_hist_data[k];
                        GHPair &src = hist_data[m];
                        float_type factor = (client_high - bin_low) / (client_high - client_low);
                        dest.g += src.g * factor;
                        dest.h += src.h * factor;
                    }
                }
            }
        }
    }
    last_hist.resize(n_bins);
    last_hist.copy_from(merged_hist);
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