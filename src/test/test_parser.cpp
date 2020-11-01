#include "gtest/gtest.h"
#include "FedTree/parser.h"
#include "FedTree/dataset.h"
#include "FedTree/Tree/tree.h"

class ParserTest: public ::testing::Test {
public:
    FLParam fl_param;
    GBDTParam param;
    vector<float_type> csr_val;
    vector<int> csr_row_ptr;
    vector<int> csr_col_idx;
    vector<float_type> y;
    size_t n_features_;
    vector<float_type> label;
protected:
    void SetUp() override {
        fl_param.mode = "horizontal";
        fl_param.n_parties = 2;
        fl_param.privacy_tech = "he";

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
        param.num_class = 1;
        param.path = DATASET_DIR "test_dataset.txt";
        param.tree_method = "auto";
        if (!param.verbose) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "True");
        }
        if (!param.profiling) {
            el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
        }
        fl_param.gbdt_param = param;
    }
};

TEST_F(ParserTest, test_parser){
    EXPECT_EQ(fl_param.gbdt_param.depth, 6);
    EXPECT_EQ(fl_param.gbdt_param.gamma, 1);
    EXPECT_EQ(fl_param.gbdt_param.learning_rate, 1);
    EXPECT_EQ(fl_param.gbdt_param.num_class, 1);
    EXPECT_EQ(fl_param.gbdt_param.tree_method, "auto");
    EXPECT_EQ(fl_param.gbdt_param.max_num_bin, 255);
}

TEST_F(ParserTest, test_save_model) {
    string model_path = "tgbm.model";
    vector<vector<Tree>> boosted_model;
    DataSet dataset;
    dataset.load_from_file(fl_param.gbdt_param.path, fl_param);
    Parser parser;
    parser.save_model(model_path, fl_param.gbdt_param, boosted_model, dataset);
}

TEST_F(ParserTest, test_load_model) {
    string model_path = "tgbm.model";
    vector<vector<Tree>> boosted_model;
    DataSet dataset;
    dataset.load_from_file(fl_param.gbdt_param.path, fl_param);
    Parser parser;
    parser.load_model(model_path, fl_param.gbdt_param, boosted_model, dataset);
    // the size of he boosted model should be zero
    EXPECT_EQ(boosted_model.size(), 0);
}
