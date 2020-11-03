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

TEST_F(HETest, test_pybind11){
    he.test();
}

//TEST_F(HETest, encryption){
//    float data = 0.01;
//    AdditivelyHE::EncryptedNumber encrypted = he.encrypt(publicKey, data);
//    float decrypt = he.decrypt(privateKey, encrypted);
//    EXPECT_EQ(data, decrypt);
//}