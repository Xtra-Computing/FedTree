#include "FedTree/Encryption/paillier_gpu.h"
#include "cgbn/cgbn.h"
#include "gmp.h"



void encrypt(SyncArray<GHPair> &message){
    auto raw_data = message.device_data();
    cgbn_error_report_t *report;
    CUDA_CHECK(cgbn_error_report_alloc(&report));
    gmp_randstate_t state = new gmp_randstate_t();
    gmp_randinit_mt(state);
    gmp_randseed_ui(state, 1000U);
    mpz_t r;
    mpz_urandomm(r, state, keylength);
    mpz_mod(r, r, modulus);

    cgbn_mem_t<keylength> random;

    from_mpz(r, random._limbs, keylength/32);

    cgbn_mem_t<keylength> m;


    device_loop(raw.size(), [=] __device__(int idx){
        context_t bn_context(cgbn_report_monitor, report, idx);
        env_t bn_env(bn_context.env<env_t>());
        env_t::cgbn_t g, m, r, n;
        cgbn_load(bn_env, g, &);


        gmp_lib.gmp_randclear(state);

        env_t::cgbn_t ;

    })
}