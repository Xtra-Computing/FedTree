//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_GBDTPARAM_H
#define FEDTREE_GBDTPARAM_H

#include <string>
#include <FedTree/common.h>

// Todo: gbdt params, refer to ThunderGBM https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/common.h
struct GBDTParam {
public:
    GBDTParam() {
        depth = 6;
        n_trees = 40;
        min_child_weight = 1;
        lambda = 1;
        gamma = 1;
        rt_eps = 1e-6;
        column_sampling_rate = 1;
        path = "../dataset/test_dataset.txt";
        //test_path = "../dataset/test_dataset.txt";
        test_path = "";
        verbose = 1;
        profiling = false;
        bagging = false;
        n_parallel_trees = 1;
        learning_rate = 1;
        objective = "reg:linear";
        num_class = 1;
        tree_per_rounds = 1; // # tree of each round, depends on # class
        max_num_bin = 255;
        n_device = 1;
        tree_method = "hist";
        random_split = true;
        seed = 0;
        global_depth = 6;
    }
    int depth;
    int n_trees;
    float_type min_child_weight;
    float_type lambda;
    float_type gamma;
    float_type rt_eps;
    float column_sampling_rate;
    std::string path;
    std::string test_path;
    int verbose;
    bool profiling;
    bool bagging;
    int n_parallel_trees;
    float learning_rate;
    std::string objective;
    int num_class;
    int tree_per_rounds; // #tree of each round, depends on #class

    //for histogram
    int max_num_bin;

    int n_device;

    std::string tree_method;

    bool random_split; // whether or not use random split for empth feature node.
    int seed; // seed for random numbers;
    int global_depth; // the depth of the global tree using in hybrid fl.
};

#endif //FEDTREE_GBDTPARAM_H
