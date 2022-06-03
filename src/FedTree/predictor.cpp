//
// Created by Kelly Yung on 2020/12/3. Code taken reference from ThunderGBM/predictor.cu
//

#include "FedTree/predictor.h"
//#include "FedTree/util/device_lambda.h"
#include "FedTree/objective/objective_function.h"

void Predictor::get_y_predict (const GBDTParam &model_param, const vector<vector<Tree>> &boosted_model,
                               const DataSet &dataSet, SyncArray<float_type> &y_predict) {
    int n_instances = dataSet.n_instances();
    int n_features = dataSet.n_features();

    //the whole model to an array
    int num_iter = boosted_model.size();
    int num_class = boosted_model.front().size();
    int num_node = boosted_model[0][0].nodes.size();
    int total_num_node = num_iter * num_class * num_node;
    y_predict.resize(n_instances * num_class);

    SyncArray<Tree::TreeNode> model(total_num_node);
    auto model_data = model.host_data();
    int tree_cnt = 0;
    for (auto &vtree:boosted_model) {
        for (auto &t:vtree) {
            memcpy(model_data + num_node * tree_cnt, t.nodes.host_data(), sizeof(Tree::TreeNode) * num_node);
            tree_cnt++;
        }
    }

    SyncArray<int> csr_col_idx(dataSet.csr_col_idx.size());
    SyncArray<float_type> csr_val(dataSet.csr_val.size());
    SyncArray<int> csr_row_ptr(dataSet.csr_row_ptr.size());
    csr_col_idx.copy_from(dataSet.csr_col_idx.data(), dataSet.csr_col_idx.size());
    csr_val.copy_from(dataSet.csr_val.data(), dataSet.csr_val.size());
    csr_row_ptr.copy_from(dataSet.csr_row_ptr.data(), dataSet.csr_row_ptr.size());

    //do prediction
    auto model_host_data = model.host_data();
    auto predict_data = y_predict.host_data();
    auto csr_col_idx_data = csr_col_idx.host_data();
    auto csr_val_data = csr_val.host_data();
    auto csr_row_ptr_data = csr_row_ptr.host_data();
    auto lr = model_param.learning_rate;

    //use sparse format and binary search
    for (int iid = 0; iid < n_instances; iid++) {

        auto get_next_child = [&](Tree::TreeNode node, float_type feaValue) {
            return feaValue < node.split_value ? node.lch_index : node.rch_index;
        };

        auto get_val = [&](const int *row_idx, const float_type *row_val, int row_len, int idx,
                           bool *is_missing) -> float_type {
            //binary search to get feature value
            const int *left = row_idx;
            const int *right = row_idx + row_len;

            while (left != right) {
                const int *mid = left + (right - left) / 2;
                if (*mid == idx) {
                    *is_missing = false;
                    return row_val[mid - row_idx];
                }
                if (*mid > idx)
                    right = mid;
                else left = mid + 1;
            }
            *is_missing = true;
            return 0;
        };

        int *col_idx = csr_col_idx_data + csr_row_ptr_data[iid];
        float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
        int row_len = csr_row_ptr_data[iid + 1] - csr_row_ptr_data[iid];
        for (int t = 0; t < num_class; t++) {
            auto predict_data_class = predict_data + t * n_instances;
            float_type sum = 0;
            for (int iter = 0; iter < num_iter; iter++) {
                const Tree::TreeNode *node_data = model_host_data + iter * num_class * num_node + t * num_node;
                Tree::TreeNode curNode = node_data[0];
                int cur_nid = 0; //node id
                while (!curNode.is_leaf) {
                    int fid = curNode.split_feature_id;
                    bool is_missing;
                    float_type fval = get_val(col_idx, row_val, row_len, fid, &is_missing);
                    if (!is_missing)
                        cur_nid = get_next_child(curNode, fval);
                    else if (curNode.default_right)
                        cur_nid = curNode.rch_index;
                    else
                        cur_nid = curNode.lch_index;
                    curNode = node_data[cur_nid];
                }
                sum += lr * node_data[cur_nid].base_weight;
            }
            predict_data_class[iid] += sum;
        }
    }
}