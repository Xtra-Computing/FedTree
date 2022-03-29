#ifndef FEDTREE_PAILLIER_GMP_H
#define FEDTREE_PAILLIER_GMP_H

#include <gmp.h>

class Paillier_GMP {
public:
    Paillier_GMP();

    Paillier_GMP& operator=(Paillier_GMP source) {
        mpz_set(this->n,source.n);
        mpz_set(this->generator, source.generator);
        this->key_length = source.key_length;
        return *this;
    }
    void keyGen(long keyLength);

    void keygen();

    void encrypt(mpz_t &r, const mpz_t &message) const;

    void decrypt(mpz_t &r, const mpz_t &ciphertext) const;

    void add(mpz_t &r, const mpz_t &x, const mpz_t &y) const;

    void mul(mpz_t &r, const mpz_t &x, const mpz_t &y) const;


    mpz_t n;
    mpz_t n_square;
    mpz_t generator;
    uint32_t key_length;


    mpz_t p, q;
    mpz_t lambda;
    mpz_t mu;

    void L_function(mpz_t &r, mpz_t &input, const mpz_t &n);
};





#endif // FEDTREE_PAILLIER_GMP_H