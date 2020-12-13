////
//// Created by Kelly Yung on 2020/12/8.
////
//
//#include "gtest/gtest.h"
//#include "FedTree/FL/FLtrainer.h"
//#include "FedTree/predictor.h"
//#include "FedTree/dataset.h"
//#include "FedTree/Tree/tree.h"
//
//class GBDTTest: public ::testing::Test {
//public:
//    GBDTParam param;
//    FLParam flParam;
//
//protected:
//    void SetUp() override {
//        // set GBDTParam
//        param.depth = 0;
//        param.n_trees = 5;
//        param.n_device = 1;
//        param.min_child_weight = 1;
//        param.lambda = 1;
//        param.gamma = 1;
//        param.rt_eps = 1e-6;
//        param.max_num_bin = 255;
//        param.verbose = false;
//        param.profiling = false;
//        param.column_sampling_rate = 1;
//        param.bagging = false;
//        param.n_parallel_trees = 1;
//        param.learning_rate = 1;
//        param.objective = "reg:linear";
//        param.num_class = 1;
//        param.path = "../dataset/test_dataset.txt";
//        param.tree_method = "hist";
//        param.tree_per_rounds = 1;
//        if (!param.verbose) {
//            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
//            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
//            el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "True");
//        }
//        if (!param.profiling) {
//            el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
//        }
//        // set FLParam
//        flParam.gbdt_param = param;
//        flParam.n_parties = 1;
//        flParam.mode = "horizontal";
//        flParam.privacy_tech = "none";
//    }
//};
//
//// test find split
//TEST_F (GBDTTest, test_find_split) {
//
//    SyncArray<SplitPoint> sp;
//    SyncArray<int_float> best_idx_gain;
//    // assume first level of tree
//    int n_nodes_in_level = 1;
//    int nid_offset = 0;
//    HistCut cut;
//    SyncArray<GHPair> hist;
//
//    // construct tree with single node
//    Tree tree;
//    SyncArray<Tree::TreeNode> nodes;
//    Tree::TreeNode node;
//    node.isValid = true;
//    auto nodesArray = nodes.host_data();
//    nodesArray[0] = node;
//
//    // instantiate HistCut and num_bins
//    HisCut cut;
//    SyncArray<float_type> cut_points_val;
//
//    SyncArray<int> cut_row_ptr;
//    SyncArray<int> cut_fid;
//    int n_bins = 2;
//    SyncArray<GHPair> hist;
//
//    //test find_split
//    find_split(sp, n_nodes_in_level, tree, best_idx_gain, nid_offset, cut, hist, n_bins);
//
//    // verify
//    EXPECT_EQ(sp.size(), n_nodes_in_level);
//}
//
//// test update_tree
//TEST_F (GBDTTest, test_update_tree) {
//
//}
//
//
//// test predictor
//TEST_F(GBDTTest, test_predictor) {
//    vector<vector<Tree>> boosted_model;
//    // construct a vector of vector of tree!
//    Tree tree;
//    SyncArray<Tree::TreeNode> nodes;
//
//    Tree::TreeNode node;
//
//    //test
//    DataSet test_dataset;
//    test_dataset.load_from_file(flParam.gbdt_param.path, flParam);
//    Predictor predictor;
//    SyncArray<float_type> y_predict;
//    predictor.get_y_predict(param, boosted_model, test_dataset, y_predict);
//}