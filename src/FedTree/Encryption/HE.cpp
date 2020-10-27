//
// Created by Kelly Yung on 2020/10/27.
//

#include "FedTree/Encryption/HE.h"

class HE {
public:
    py::module paillier;

    struct PaillierPublicKey {
        py::object publickey;
    };

    struct PaillierPrivateKey {
        py::object privatekey;
    };

    struct EncryptedNumber {
        PaillierPublicKey publicKey,
        int ciphertext,
        int exponent
    };

    HE() {
        // import the python phe library
        py::module paillier = py::module_::import("phe");
    }

    // generate key pairs
    std::tuple<PaillierPublicKey, PaillierPrivateKey> generate_key_pairs () {
        std::tuple<py::object, py::object> key_pairs = os.attr("generate_paillier_keypair").cast<std:tuple<py::object, py::object>>();
        PaillierPublicKey publicKey {std::get<0>(key_pairs)};
        PaillierPrivateKey privateKey {std::get<1>(key_pairs)};
        std::tuple<PaillierPublicKey, PaillierPrivateKey> wrapped_key_pairs = std::make_tuple(publicKey, privateKey);
        return wrapped_key_pairs;
    }

    // encrypt with public key and return EncryptedNumber (pyobject)
    EncryptedNumber encrypt(PaillierPublicKey publicKey, float value) {
        py::object encrypted = publicKey.publickey.attr("encrypt")(value);
        EncryptedNumber wrapped_encryped = EncryptedNumber num {publicKey, encrypted.attr("ciphertext").cast<int>(), encrypted.attr("exponent").cast<int>()};
        return wrapped_encryped;
    }

    // decrypt with private key and return EncryptedNumber (pyobject)
    float decrypt(PaillierPrivateKey privateKey, EncryptedNumber encrypted_value) {
        py::object encrypted = paillier.attr("EncrypedNumber")(encrypted_value.publicKey.publickey, encrypted_value.ciphertext, encrypted_value.exponent);
        float decrypted = privateKey.privatekey.attr("decrypt")(encrypted).cast<float>();
        return decrypted
    }

    // aggregate encrypted numbers
    EncryptedNumber aggregate(EncryptedNumber encrypted_number1, EncryptedNumber encrypted_number2) {
        py::object unwrapped_number1 = paillier.attr("EncrypedNumber")(encrypted_number1.publicKey.publickey, encrypted_number1.ciphertext, encrypted_number1.exponent);
        py::object unwrapped_number2 = paillier.attr("EncrypedNumber")( encrypted_number2.publicKey.publickey,  encrypted_number2.ciphertext,  encrypted_number2.exponent);
        py::object aggregation = unwrapped_number1.attr(__add__)(unwrapped_number2);
        EncryptedNumber wrapped = EncryptedNumber aggregate {encrypted_number1.publicKey, aggregation.attr("ciphertext").cast<int>(), aggregation.attr("exponent").cast<int>()};
        return wrapped
    }