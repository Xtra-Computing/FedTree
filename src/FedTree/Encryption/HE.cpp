//
// Created by Kelly Yung on 2020/10/27.
//

#include "FedTree/Encryption/HE.h"

py::scoped_interpreter guard{};

AdditivelyHE::AdditivelyHE () {
    paillier = py::module_::import("phe");
}

void AdditivelyHE::test() {
    py::print("Hello World!");
}

// generate key pairs
std::tuple <AdditivelyHE::PaillierPublicKey, AdditivelyHE::PaillierPrivateKey> AdditivelyHE::generate_key_pairs() {
    std::tuple < py::object, py::object > key_pairs = paillier.attr("generate_paillier_keypair").cast < std::tuple <py::object,
            py::object >> ();
    PaillierPublicKey publicKey{std::get<0>(key_pairs)};
    PaillierPrivateKey privateKey{std::get<1>(key_pairs)};
    std::tuple <PaillierPublicKey, PaillierPrivateKey> wrapped_key_pairs = std::make_tuple(publicKey, privateKey);
    return wrapped_key_pairs;
}

AdditivelyHE::EncryptedNumber AdditivelyHE::encrypt(AdditivelyHE::PaillierPublicKey publicKey, float value) {
    py::object encrypted = publicKey.publickey.attr("encrypt")(value);
    EncryptedNumber wrapped_encryped = {publicKey, encrypted.attr("ciphertext").cast<int>(), encrypted.attr("exponent").cast<int>()};
    return wrapped_encryped;
}

// decrypt with private key and return EncryptedNumber (pyobject)
float AdditivelyHE::decrypt(PaillierPrivateKey privateKey, AdditivelyHE::EncryptedNumber encrypted_value) {
    py::object encrypted = paillier.attr("EncrypedNumber")(encrypted_value.publicKey.publickey,
                                                           encrypted_value.ciphertext, encrypted_value.exponent);
    float decrypted = paillier.attr("decrypt")(encrypted).cast<float>();
    return decrypted;
}

// aggregate encrypted numbers
AdditivelyHE::EncryptedNumber AdditivelyHE::aggregate(AdditivelyHE::EncryptedNumber encrypted_number1, AdditivelyHE::EncryptedNumber encrypted_number2) {
    py::object unwrapped_number1 = paillier.attr("EncrypedNumber")(encrypted_number1.publicKey.publickey,
                                                                   encrypted_number1.ciphertext,
                                                                   encrypted_number1.exponent);
    py::object unwrapped_number2 = paillier.attr("EncrypedNumber")(encrypted_number2.publicKey.publickey,
                                                                   encrypted_number2.ciphertext,
                                                                   encrypted_number2.exponent);
    py::object aggregation = unwrapped_number1.attr("__add__")(unwrapped_number2);
    AdditivelyHE::EncryptedNumber wrapped = {encrypted_number1.publicKey, aggregation.attr("ciphertext").cast<int>(),
              aggregation.attr("exponent").cast<int>()};
    return wrapped;
}