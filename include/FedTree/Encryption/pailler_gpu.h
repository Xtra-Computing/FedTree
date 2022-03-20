#include

class Paillier {
public:
    Paillier();

    explicit Paillier(long keyLength);

    void encrypt(SyncArray<GHPair> &message) const;

    void decrypt(SyncArray<GHPair> &ciphertext) const;

    NTL::ZZ add(SyncArray<GHPair> &x, SyncArray<GHPair> &y) const;

    NTL::ZZ mul(SyncArray<GHPair> &x, SyncArray<GHPair> &y) const;


    cgbn_mem_t<512> modulus;
    mpz_t modulus;
    cgbn_mem_t<512> generator;
    long keyLength;

private:
    cgbn_mem_t<512> p, q;
    cgbn_mem_t<512> lambda;
    cgbn_mem_t<512> lambda_power;
    cgbn_mem_t<512> u;

    NTL::ZZ L_function(const NTL::ZZ &n) const { return (n - 1) / modulus; }
};

