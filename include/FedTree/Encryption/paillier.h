#pragma once

#include <NTL/ZZ.h>
#include <NTL/ZZ_pXFactoring.h>

class Paillier {
public:
    Paillier();

    Paillier& operator=(Paillier source) {
        this->modulus = source.modulus;
        this->generator = source.generator;
        this->keyLength = source.keyLength;

//        this->random = source.random;
        return *this;
    }
    void keygen(long keyLength);

    NTL::ZZ encrypt(const NTL::ZZ &message) const;

    NTL::ZZ decrypt(const NTL::ZZ &ciphertext) const;

    NTL::ZZ add(const NTL::ZZ &x, const NTL::ZZ &y) const;

    NTL::ZZ mul(const NTL::ZZ &x, const NTL::ZZ &y) const;

    NTL::ZZ modulus;
    NTL::ZZ generator;
    long keyLength;

//private:
    NTL::ZZ p, q;
    NTL::ZZ lambda;
    NTL::ZZ lambda_power;
    NTL::ZZ u;

//    NTL::ZZ random;

    NTL::ZZ L_function(const NTL::ZZ &n) const { return (n - 1) / modulus; }
};

