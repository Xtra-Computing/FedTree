//
// Created by Kelly Yung on 2020/10/27.
//

#include "FedTree/Encryption/HE.h"

py::scoped_interpreter guard{};

AdditivelyHE::AdditivelyHE () {
    paillier = py::module_::import("phe");
}

// generate key pairs
std::tuple <AdditivelyHE::PaillierPublicKey, AdditivelyHE::PaillierPrivateKey> AdditivelyHE::generate_key_pairs() {
    py::tuple key_pairs = paillier.attr("generate_paillier_keypair")();
    PaillierPublicKey publicKey{key_pairs[0]};
    PaillierPrivateKey privateKey{key_pairs[1]};
    std::tuple <PaillierPublicKey, PaillierPrivateKey> wrapped_key_pairs = std::make_tuple(publicKey, privateKey);
    return wrapped_key_pairs;
}

AdditivelyHE::EncryptedNumber AdditivelyHE::encrypt(AdditivelyHE::PaillierPublicKey publicKey, float value) {
    py::object encrypted = publicKey.publickey.attr("encrypt")(py::float_(value));
    EncryptedNumber wrapped_encrypted = {encrypted};
    return wrapped_encrypted;
}

// decrypt with private key and return EncryptedNumber (pyobject)
float AdditivelyHE::decrypt(PaillierPrivateKey privateKey, AdditivelyHE::EncryptedNumber encrypted_value) {
    float decrypted = py::float_(privateKey.privatekey.attr("decrypt")(encrypted_value.encrypted));
    return decrypted;
}

// aggregate two encrypted numbers
AdditivelyHE::EncryptedNumber AdditivelyHE::aggregate(AdditivelyHE::EncryptedNumber encrypted_number1, AdditivelyHE::EncryptedNumber encrypted_number2) {
    py::object aggregation = encrypted_number1.encrypted.attr("__add__")(encrypted_number2.encrypted);
    AdditivelyHE::EncryptedNumber wrapped = {aggregation};
    return wrapped;
}

// aggregate encrypted numbers with scalar number
AdditivelyHE::EncryptedNumber AdditivelyHE::aggregate_scalar(AdditivelyHE::EncryptedNumber encrypted_number1, float scalar) {
    py::object aggregation = encrypted_number1.encrypted.attr("__add__")(py::float_(scalar));
    AdditivelyHE::EncryptedNumber wrapped = {aggregation};
    return wrapped;
}

// aggregate encrypted numbers with scalar number
AdditivelyHE::EncryptedNumber AdditivelyHE::multiply_scalar(AdditivelyHE::EncryptedNumber encrypted_number1, float scalar) {
    py::object aggregation = encrypted_number1.encrypted.attr("__mul__")(py::float_(scalar));
    AdditivelyHE::EncryptedNumber wrapped = {aggregation};
    return wrapped;
}