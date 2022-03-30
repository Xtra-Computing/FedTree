#ifndef FEDTREE_PAILLIER_GPU_CUH
#define FEDTREE_PAILLIER_GPU_CUH


#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "FedTree/syncarray.h"
#include "FedTree/common.h"
#include "FedTree/Encryption/paillier_gmp.h"

#define TPI 32
#define ENV_BITS 1024
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, ENV_BITS> env_t;

template<uint32_t BITS>
class cgbn_gh{
public:
    cgbn_mem_t<BITS> g;
    cgbn_mem_t<BITS> h;
};

template<uint32_t BITS>
class Paillier_GPU {
public:
    Paillier_GPU(): key_length(BITS) {};


//    Paillier_GPU& operator=(Paillier_GPU source) {
//        this->paillier_cpu = source.paillier_cpu;
//        this->generator = source.generator;
//        this->keyLength = source.keyLength;
//        return *this;
//    }

    void keygen();
//    explicit Paillier_GPU(unit32_t key_length);
    void L_function(mpz_t result, mpz_t input, mpz_t N);

    void encrypt(SyncArray<GHPair> &message);

    void encrypt(GHPair &message);

    void decrypt(SyncArray<GHPair> &ciphertext);

    void decrypt(GHPair &message);

    void add(mpz_t &result, mpz_t &x, mpz_t &y);
    void mul(mpz_t result, mpz_t &x, mpz_t &y);

//    cgbn_mem_t<BITS> add(SyncArray<GHPair> &x, SyncArray<GHPair> &y) const;

//    cgbn_mem_t<BITS> mul(SyncArray<GHPair> &x, SyncArray<GHPair> &y) const;


//    cgbn_mem_t<BITS> modulus;
//    cgbn_mem_t<BITS> generator;

//    mpz_t n;
//    mpz_t n_square;
//    mpz_t generator;
    uint32_t key_length;

    cgbn_mem_t<BITS> *n_gpu;
    cgbn_mem_t<BITS> *n_square_gpu;
    cgbn_mem_t<BITS> *generator_gpu;

//    cgbn_gh_results<BITS>* gh_results_gpu;

    Paillier_GMP paillier_cpu;

private:
//    mpz_t p, q;
//    mpz_t lambda;

//    mpz_t mu;

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



#endif //FEDTREE_PAILLIER_GPU_CUH