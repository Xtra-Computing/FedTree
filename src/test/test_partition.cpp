//
// Created by hanyuxuan on 22/10/20.
//

#include "gtest/gtest.h"
#include "FedTree/dataset.h"
#include "FedTree/FL/FLparam.h"
#include "FedTree/FL/partition.h"


class PartitionTest : public ::testing::Test {
public:
    FLParam fl_param;
    DataSet dataset;
protected:
    void SetUp() override {
        dataset.load_from_file(DATASET_DIR
        "test_dataset.txt", fl_param);
    }
};

TEST_F(PartitionTest, homo_partition) {
    printf("### Dataset: test_dataset.txt, num_instances: %zu, num_features: %zu, get_cut_points finished. ###\n",
           dataset.n_instances(),
           dataset.n_features());
    EXPECT_EQ(dataset.n_instances(), 1605);
    EXPECT_EQ(dataset.n_features_, 119);
    EXPECT_EQ(dataset.label[0], -1);
    EXPECT_EQ(dataset.csr_val[1], 1);

    printf("### Test Partition ###\n");
    Partition partition;
    // TODO
//        std::map<int, vector<int>> batch_idxs = partition.homo_partition(dataset, 5, true);
//    EXPECT_EQ(batch_idxs.size(), 5);
//    int count = 0;
//    for (auto const &x : batch_idxs) EXPECT_EQ(x.second.size(), 1605/5);
}

TEST_F(PartitionTest, hetero_partition) {
    printf("### Dataset: test_dataset.txt, num_instances: %zu, num_features: %zu, get_cut_points finished. ###\n",
           dataset.n_instances(),
           dataset.n_features());
    EXPECT_EQ(dataset.n_instances(), 1605);
    EXPECT_EQ(dataset.n_features_, 119);
    EXPECT_EQ(dataset.label[0], -1);
    EXPECT_EQ(dataset.csr_val[1], 1);

    printf("### Test Partition ###\n");
    Partition partition;
    // TODO: test values of subsets
    // vector<DataSet> subsets(5); 
    // std::map<int, vector<int>> batch_idxs = partition.hetero_partition(dataset, 5, false, subsets);
    // EXPECT_EQ(batch_idxs.size(), 5);
    // int count = 0;
    // for (auto const &x : batch_idxs) count += x.second.size();
    // EXPECT_EQ(count, dataset.n_features_);
}
