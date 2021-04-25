//
// Created by Kelly Yung on 2020/12/9.
//

#include "gtest/gtest.h"
#include "FedTree/objective/regression_obj.h"
#include "FedTree/dataset.h"
#include "FedTree/syncarray.h"

class GradientTest: public ::testing::Test {
public:
    FLParam fl_param;
    GBDTParam param;
protected:
    void SetUp() override {
        param.depth = 6;
        param.n_trees = 40;
        param.n_device = 1;
        param.min_child_weight = 1;
        param.lambda = 1;
        param.gamma = 1;
        param.rt_eps = 1e-6;
        param.max_num_bin = 255;
        param.verbose = false;
        param.profiling = false;
        param.column_sampling_rate = 1;
        param.bagging = false;
        param.n_parallel_trees = 1;
        param.learning_rate = 1;
        param.objective = "reg:linear";
        param.num_class = 2;
        param.path = "../dataset/test_dataset.txt";
        param.tree_method = "hist";
        if (!param.verbose) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "True");
        }
        if (!param.profiling) {
            el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
        }
        fl_param.gbdt_param = param;
        fl_param.privacy_tech = "none";
        fl_param.mode = "horizontal";
        fl_param.n_parties = 1;
    }
};

// Test regression function
TEST_F(GradientTest, test_regression_obj) {
    DataSet dataset;
    dataset.load_from_file(param.path, fl_param);
    RegressionObj<SquareLoss> rmse;
    SyncArray<float_type> y_true(4);
    SyncArray<float_type> y_pred(4);
    auto y_pred_data = y_pred.host_data();
    for(int i = 0; i < 4; i++)
        y_pred_data[i] = -i;
    SyncArray<GHPair> gh_pair(4);
    EXPECT_EQ(rmse.default_metric_name(), "rmse");
    rmse.get_gradient(y_true, y_pred, gh_pair);
    auto gh_pair_data = gh_pair.host_data();
    EXPECT_EQ(gh_pair_data[0], GHPair(0.0, 1.0));
    EXPECT_EQ(gh_pair_data[1], GHPair(-1.0, 1.0));
    EXPECT_EQ(gh_pair_data[2], GHPair(-2.0, 1.0));
    EXPECT_EQ(gh_pair_data[3], GHPair(-3.0, 1.0));
}

TEST_F(GradientTest, test_logcls_obj) {
    DataSet dataset;
    dataset.load_from_file(param.path, fl_param);
    LogClsObj<SquareLoss> logcls;
    SyncArray<float_type> y_true(4);
    SyncArray<float_type> y_pred(4);
    auto y_pred_data = y_pred.host_data();
    for(int i = 0; i < 4; i++)
        y_pred_data[i] = -i;
    SyncArray<GHPair> gh_pair(4);
    EXPECT_EQ(logcls.default_metric_name(), "error");
    logcls.get_gradient(y_true, y_pred, gh_pair);
    std::cout << gh_pair;
    auto gh_pair_data = gh_pair.host_data();
    EXPECT_EQ(gh_pair_data[0], GHPair(0.0, 1.0));
    EXPECT_EQ(gh_pair_data[1], GHPair(-1.0, 1.0));
    EXPECT_EQ(gh_pair_data[2], GHPair(-2.0, 1.0));
    EXPECT_EQ(gh_pair_data[3], GHPair(-3.0, 1.0));
}