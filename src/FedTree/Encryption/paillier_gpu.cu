#include "FedTree/Encryption/paillier_gpu.h"


cgbn_mem_t<BITS> add(SyncArray<GHPair> &x, SyncArray<GHPair> &y){

}

void encrypt(SyncArray<GHPair> &message){
    auto raw_data = message.device_data();
    cgbn_error_report_t *report;
    CUDA_CHECK(cgbn_error_report_alloc(&report));
    gmp_randstate_t state = new gmp_randstate_t();
    gmp_randinit_mt(state);
    gmp_randseed_ui(state, 1000U);
    mpz_t r;
    mpz_urandomm(r, state, keylength);
    mpz_mod(r, r, modulus_cpu);

    cgbn_mem_t<keylength> random;
    from_mpz(r, random._limbs, keylength/32);
    cgbn_mem_t<keylength> message_gpu;

    // todo: move values to gpu first
    device_loop(message.size(), [=] __device__(int idx){
        context_t bn_context(cgbn_report_monitor, report, idx);
        env_t bn_env(bn_context.env<env_t>());
        env_t::cgbn_t g, m, r, n, n_square, re1, re2, result;
        env_t::cgbn_wide_t w;
        cgbn_set_ui32(bn_env, m, (uint32_t) (raw_data[idx].g * 1e6));

        cgbn_load(bn_env, g, &generator);
        cgbn_load(bn_env, r, &random);
        cgbn_load(bn_env, n, &modulus);

        cgbn_sqr(bn_env, n_square, n);
        // todo: check the bit of n, should be smaller than half BITS, set n to 512 bits and use BITS=1024

        // compute g_enc
        cgbn_modular_power(bn_env, re1, g, m, n_square);
        cgbn_modular_power(bn_env, re2, r, n, n_square);
        cgbn_mul_wide(bn_env, w, re1, re2);
        cgbn_rem_wide(bn_env, result, w, n_square);
        cgbn_store(bn_env, &(raw_data[idx].g_enc), result);

        // compute h_enc
        cgbn_set_ui32(bn_env, m, (uint32_t) (raw_data[idx].h * 1e6));
        cgbn_modular_power(bn_env, re1, g, m, n_square);
        cgbn_mul_wide(bn_env, w, re1, re2);
        cgbn_rem_wide(bn_env, result, w, n_square);
        cgbn_store(bn_env, &(raw_data[idx].h_enc), result);
    });
}

void decrypt(SyncArray<GHPair> &message){
    auto raw_data = message.device_data();
    cgbn_error_report_t *report;
    CUDA_CHECK(cgbn_error_report_alloc(&report));
    device_loop(message.size(), [=] __device__(int idx){
        context_t bn_context(cgbn_report_monitor, report, idx);
        env_t bn_env(bn_context.env<env_t>());
        env_t::cgbn_t c, lambda, n, mu, n_square, re1, re2, re3, re4;
        cgbn_load(bn_env, c, &raw_data[idx].g_enc);
        cgbn_load(bn_env, lambda, &this->lambda);
        cgbn_load(bn_env, n, &modulus);
        cgbn_sqr(bn_env, n_square, n);

        cgbn_modular_power(bn_env, re1, c, lambda, n_square);
        cgbn_sub_ui32(bn_env, re2, re1, 1);
        cgbn_div(bn_env, re3, re2, n);

        cgbn_mul(bn_env, re4, re3, mu);
        cgbn_rem(bn_env, result, re4, n);

        raw_data[idx].g = (float_type) cgbn_get_ui32(bn_env, result) / 1e6;
        // todo: check whether cpu is syncrhonized or not

        cgbn_load(bn_env, c, &raw_data[idx].h_enc);
        cgbn_modular_power(bn_env, re1, c, lambda, n_square);
        cgbn_sub_ui32(bn_env, re2, re1, 1);
        cgbn_div(bn_env, re3, re2, n);
        cgbn_mul(bn_env, re4, re3, mu);
        cgbn_rem(bn_env, result, re4, n);
        raw_data[idx].h = (float_type) cgbn_get_ui32(bn_env, result) / 1e6;
    });
}