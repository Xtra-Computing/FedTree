#include "cgbn/cgbn.h"
#include <gmp.h>


template<uint32_t BITS>
class cgbn_gh_results{
public:
    cgbn_mem_t<BITS> g;
    cgbn_mem_t<BITS> h;
};

template<uint32_t BITS>
class Paillier_GPU {
public:
    Paillier_GPU() key_length(BITS) {mpz_init(n); mpz_init(n_square); mpz_init(generator); mpz_init(lambda);
        mpz_init(p); mpz_init(q); mpz_init(mu);};


    Paillier_GPU& operator=(Paillier_GPU source) {
        this->n = source.n;
        this->generator = source.generator;
        this->keyLength = source.keyLength;
    }

    void keygen();
//    explicit Paillier_GPU(unit32_t key_length);
    void L_function(mpz_t result, mpz_t input, mpz_t N);

    void encrypt(SyncArray<GHPair> &message) const;

    void decrypt(SyncArray<GHPair> &ciphertext) const;

    void decrypt(GHPair &message);

    void add(mpz_t &result, mpz_t &x, mpz_t &y);
    void mul(mpz_t result, mpz_t &x, mpz_t &y);

//    cgbn_mem_t<BITS> add(SyncArray<GHPair> &x, SyncArray<GHPair> &y) const;

//    cgbn_mem_t<BITS> mul(SyncArray<GHPair> &x, SyncArray<GHPair> &y) const;


//    cgbn_mem_t<BITS> modulus;
//    cgbn_mem_t<BITS> generator;
    mpz_t n;
    mpz_t n_square;
    mpz_t generator;
    unit32_t key_length;

    cgbn_mem_t<BITS> *n_gpu;
    cgbn_mem_t<BITS> *n_square_gpu;
    cgbn_mem_t<BITS> *generator_gpu;

//    cgbn_gh_results<BITS>* gh_results_gpu;

private:
    mpz_t p, q;
    mpz_t lambda;

    mpz_t mu;

    cgbn_mem_t<BITS> *lambda_gpu;
    cgbn_mem_t<BITS> *mu_gpu;

};

template<uint32_t BITS>
class cgbn_pailler_encryption_parameters{
public:
    cgbn_mem_t<BITS> n;
    cgbn_mem_t<BITS> n_square;
    cgbn_mem_t<BITS> generator;
    cgbn_mem_t<BITS> random;
};

template<uint32_t BITS>
class cgbn_pailler_decryption_parameters{
public:
    cgbn_mem_t<BITS> n;
    cgbn_mem_t<BITS> n_square;
    cgbn_mem_t<BITS> lambda;
    cgbn_mem_t<BITS> mu;
};



