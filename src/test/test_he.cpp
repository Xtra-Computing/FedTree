//
// Created by kellyyung on 22/10/2020.
//

#include "FedTree/Encryption/HE.h"
#include "gtest/gtest.h"

class HETest : public ::testing::Test {
public:
    AdditivelyHE he;
    AdditivelyHE::PaillierPublicKey publicKey;
    AdditivelyHE::PaillierPrivateKey privateKey;

protected:
    void SetUp() override {
        std::tuple <AdditivelyHE::PaillierPublicKey, AdditivelyHE::PaillierPrivateKey> keyPairs = he.generate_key_pairs();
        publicKey = std::get<0>(keyPairs);
        privateKey = std::get<1>(keyPairs);
    }

};

TEST_F(HETest, encryption){
    float data = 0.01;
    AdditivelyHE::EncryptedNumber encrypted = he.encrypt(publicKey, data);
    float decrypt = he.decrypt(privateKey, encrypted);
    EXPECT_EQ(data, decrypt);
}

TEST_F(HETest, aggregation){
    float data1 = 0.01;
    float data2 = 0.03;
    AdditivelyHE::EncryptedNumber encrypted1 = he.encrypt(publicKey, data1);
    AdditivelyHE::EncryptedNumber encrypted2 = he.encrypt(publicKey, data2);
    AdditivelyHE::EncryptedNumber aggregate = he.aggregate(encrypted1, encrypted2);
    float decrypt = he.decrypt(privateKey, aggregate);
    EXPECT_EQ(data1+data2, decrypt);
}

TEST_F(HETest, negative_and_different_precision){
    float data1 = -0.001;
    float data2 = -0.03;
    AdditivelyHE::EncryptedNumber encrypted1 = he.encrypt(publicKey, data1);
    AdditivelyHE::EncryptedNumber encrypted2 = he.encrypt(publicKey, data2);
    AdditivelyHE::EncryptedNumber aggregate = he.aggregate(encrypted1, encrypted2);
    float decrypt = he.decrypt(privateKey, aggregate);
    EXPECT_EQ(data1+data2, decrypt);
}

// aggregate with scalar
TEST_F(HETest, aggregate_scalar){
float data1 = 0.001;
float data2 = 0.1;
AdditivelyHE::EncryptedNumber encrypted1 = he.encrypt(publicKey, data1);
AdditivelyHE::EncryptedNumber aggregate = he.aggregate_scalar(encrypted1, data2);
float decrypt = he.decrypt(privateKey, aggregate);
EXPECT_EQ(data1+data2, decrypt);
}

// multiply with scalar
TEST_F(HETest, multiply_scalar){
float data1 = 0.001;
float data2 = 1.0;
AdditivelyHE::EncryptedNumber encrypted1 = he.encrypt(publicKey, data1);
AdditivelyHE::EncryptedNumber aggregate = he.multiply_scalar(encrypted1, data2);
float decrypt = he.decrypt(privateKey, aggregate);
EXPECT_EQ(data1*data2, decrypt);
}
