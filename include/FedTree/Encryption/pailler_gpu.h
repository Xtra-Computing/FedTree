#include "cgbn/cgbn.h"
#include <gmp.h>

template<uint32_t BITS>
class Paillier {
public:
    Paillier();

    explicit Paillier();

    void encrypt(SyncArray<GHPair> &message) const;

    void decrypt(SyncArray<GHPair> &ciphertext) const;

    cgbn_mem_t<BITS> add(SyncArray<GHPair> &x, SyncArray<GHPair> &y) const;

    cgbn_mem_t<BITS> mul(SyncArray<GHPair> &x, SyncArray<GHPair> &y) const;


    cgbn_mem_t<BITS> modulus;
    mpz_t modulus_cpu;
    cgbn_mem_t<BITS> generator;
    long keyLength;

private:
    cgbn_mem_t<BITS> p, q;
    cgbn_mem_t<BITS> lambda;
    cgbn_mem_t<BITS> lambda_power;
    cgbn_mem_t<BITS> u;

    NTL::ZZ L_function(const NTL::ZZ &n) const { return (n - 1) / modulus; }
};

