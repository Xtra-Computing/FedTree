//
// Created by liqinbin on 10/19/20.
//
#include "gtest/gtest.h"
#include "FedTree/dataset.h"

class DatasetTest : public ::testing::Test {
public:
    FLParam fl_param;
    DataSet dataset;
protected:
    void SetUp() override {
        dataset.load_from_file(DATASET_DIR "test_dataset.txt", fl_param);
    }
};

class DatasetLoadCscTest : public ::testing::Test {
public:
    FLParam fl_param;
    DataSet dataset;
protected:
    void SetUp() override {
        dataset.load_csc_from_file(DATASET_DIR "test_dataset.txt", fl_param, 119);
    }
};

TEST_F(DatasetTest, load_from_file){
    printf("### Dataset: test_dataset.txt, num_instances: %zu, num_features: %zu, get_cut_points finished. ###\n",
        dataset.n_instances(),
        dataset.n_features());
    EXPECT_EQ(dataset.n_instances(), 1605);
    EXPECT_EQ(dataset.n_features_, 119);
    EXPECT_EQ(dataset.label[0], -1);
    EXPECT_EQ(dataset.csr_val[1], 1);
}

TEST_F(DatasetLoadCscTest, load_csc_from_file){
    printf("### Dataset: test_dataset.txt, num_instances: %zu, num_features: %zu, get_cut_points finished. ###\n",
           dataset.n_instances(),
           dataset.n_features());
    EXPECT_EQ(dataset.n_instances(), 1605);
    EXPECT_EQ(dataset.n_features_, 119);
    EXPECT_EQ(dataset.label[0], -1);
    EXPECT_EQ(dataset.csc_val[1], 1);
}

//TEST(DatasetTest, load_dataset){
//    DataSet dataset;
//    FLParams param;
//    dataset.load_from_file(DATASET_DIR "test_dataset.txt", param);
//    printf("### Dataset: %s, num_instances: %d, num_features: %d, get_cut_points finished. ###\n",
//    param.path.c_str(),
//            dataset.n_instances(),
//            dataset.n_features());
//    EXPECT_EQ(dataset.n_instances(), 1605);
//    EXPECT_EQ(dataset.n_features_, 119);
//    EXPECT_EQ(dataset.label[0], -1);
//    EXPECT_EQ(dataset.csr_val[1], 1);
//}
