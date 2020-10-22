//
// Created by kellyyung on 22/10/2020.
//

#include "FedTree/Encryption/HomomorphicEncrpytion.h"
#include "gtest/gtest.h"

class HETest : public ::testing::Test {

};

TEST_F(HETest, generate_key_pairs){
printf("### Dataset: test_dataset.txt, num_instances: %d, num_features: %d, get_cut_points finished. ###\n",
dataset.n_instances(),
        dataset.n_features());
EXPECT_EQ(dataset.n_instances(), 1605);
EXPECT_EQ(dataset.n_features_, 119);
EXPECT_EQ(dataset.label[0], -1);
EXPECT_EQ(dataset.csr_val[1], 1);
}