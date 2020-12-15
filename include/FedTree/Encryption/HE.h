//
// Created by kellyyung on 19/10/2020.
//

#ifndef FEDERATEDLEARNINH_HE_H
#define FEDERATEDLEARNINH_HE_H

#include "pybind11/embed.h"
#include "pybind11/cast.h"
#include <tuple>


namespace py = pybind11;
using namespace py::literals;

class AdditivelyHE {
public:
    py::module paillier;

    AdditivelyHE();

    struct PaillierPublicKey {
        py::object publickey;
    };

    struct PaillierPrivateKey {
        py::object privatekey;
    };

    struct EncryptedNumber {
        py::object encrypted;
    };

    // generate key pairs
    std::tuple<PaillierPublicKey, PaillierPrivateKey> generate_key_pairs();
    // encrypt with public key and return EncryptedNumber (pyobject)
    EncryptedNumber encrypt(PaillierPublicKey publicKey, float value);
    // decrypt with private key and return EncryptedNumber (pyobject)
    float decrypt(PaillierPrivateKey privateKey, EncryptedNumber encrypted_value);
    // aggregate encrypted numbers
    EncryptedNumber aggregate(EncryptedNumber encrypted_number1, EncryptedNumber encrypted_number2);
    // aggregate encrypted numbers with scalar
    EncryptedNumber aggregate_scalar(EncryptedNumber encrypted_number1, float scalar);
    // multiply encrypted numbers with scalar
    EncryptedNumber multiply_scalar(EncryptedNumber encrypted_number1, float scalar);
};

#endif //FEDERATEDLEARNINH_HE_H
