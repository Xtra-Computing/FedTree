#include "gtest/gtest.h"
#include "FedTree/Tree/tree_builder.h"


class TreeBuilderTest : public ::testing::Test {
public:
    GHPair father;
    GHPair lch;
    GHPair rch;
    TreeBuilder treeBuilder;
protected:
    void SetUp() override {
        father = GHPair(5.1, 5);
        lch = GHPair(-2.1, 2);
        rch = GHPair(9.61, 3);
    }
};

TEST_F(TreeBuilderTest, compute_gain) {

EXPECT_FLOAT_EQ(treeBuilder.compute_gain(father, lch, rch, 0.1), 26.791);
}
