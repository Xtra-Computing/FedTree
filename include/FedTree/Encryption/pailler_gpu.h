#include "cgbn/cgbn.h"
#include <gmp.h>

template<uint32_t BITS>
class Paillier_GPU {
public:
    Paillier_GPU() {mpz_init(n); mpz_init(n_square); mpz_init()};

    Paillier_GPU& operator=(Paillier_GPU source) {
        this->n = source.n;
        this->generator = source.generator;
        this->keyLength = source.keyLength;
    }

    explicit Paillier_GPU(unit32_t key_length);
    void L_function(mpz_t result, mpz_t input, mpz_t N);

    void encrypt(SyncArray<GHPair> &message) const;

    void decrypt(SyncArray<GHPair> &ciphertext) const;

    void add(mpz_t &result, mpz_t &x, mpz_t &y);
    void mul(mpz_t result, mpz_t &x, mpz_t &y);

//    cgbn_mem_t<BITS> add(SyncArray<GHPair> &x, SyncArray<GHPair> &y) const;

//    cgbn_mem_t<BITS> mul(SyncArray<GHPair> &x, SyncArray<GHPair> &y) const;


    cgbn_mem_t<BITS> modulus;
    cgbn_mem_t<BITS> generator;
    mpz_t n;
    mpz_t n_square;
    mpz_t generator;
    long keyLength;

private:
    cgbn_mem_t<BITS> p, q;
    cgbn_mem_t<BITS> lambda;
    cgbn_mem_t<BITS> lambda_power;
    cgbn_mem_t<BITS> u;

};

