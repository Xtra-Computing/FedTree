#include

template<uint32_t BITS>
class Paillier {
public:
    Paillier();

    explicit Paillier();

    void encrypt(SyncArray<GHPair> &message) const;

    void decrypt(SyncArray<GHPair> &ciphertext) const;

    NTL::ZZ add(SyncArray<GHPair> &x, SyncArray<GHPair> &y) const;

    NTL::ZZ mul(SyncArray<GHPair> &x, SyncArray<GHPair> &y) const;


    cgbn_mem_t<BITS> modulus;
    mpz_t modulus;
    cgbn_mem_t<BITS> generator;
    long keyLength;

private:
    cgbn_mem_t<BITS> p, q;
    cgbn_mem_t<BITS> lambda;
    cgbn_mem_t<BITS> lambda_power;
    cgbn_mem_t<BITS> u;

    NTL::ZZ L_function(const NTL::ZZ &n) const { return (n - 1) / modulus; }
};

