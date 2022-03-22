#include "FedTree/Encryption/paillier_gpu.h"


void Paillier_GPU::add(mpz_t &result, mpz_t &x, mpz_t &y){
    mpz_mul(result, x, y);
    mpz_mod(result, result, n_square);
    return;
}

void Paillier_GPU::mul(mpz_t result, mpz_t &x, mpz_t &y){
    mpz_powm(result, x, y, n_square);
    return ;
}

void Paillier_GPU::L_function(mpz_t result, mpz_t input, mpz_t N){
    mpz_sub_ui(result, input, 1);
    mpz_tdiv_q(result, result, N);
}

void Paillier_GPU::keygen(long key_length){
    gmp_randstate_t state = new gmp_randstate_t();
    gmp_randinit_mt(state);
    gmp_randseed_ui(state, 1000U);
    mpz tmp1, tmp2, tmp3, tmp4;
    mpz_init(tmp1);
    mpz_init(tmp2);
    mpz_init(tmp3);
    mpz_init(tmp4);
    while (true){
        mpz_urandomb(p, gpc_randstate, key_length/4);
        mpz_urandomb(q, gpc_randstate, key_length/4);
        mpz_nextprime(p, p);
        mpz_nextprime(q, q);
        mpz_sub_ui(tmp1, p, 1);
        mpz_sub_ui(tmp2, q, 1);
        mpz_mul(tmp3, tmp1, tmp2); // tmp3 = (p-1)(q-1)
        mpz_mul(tmp4, p, q); // tmp4 = p*q
        mpz_gcd(tmp3, tmp3, tmp4); // tmp = gcd(pq, (p-1)(q-1))
        if(mpz_cmp_ui(tmp3, 1) == 0) // gcd(pq, (p-1)(q-1)) == 1
            break;
    }

    n = tmp4;                                                        // n = p * q
    mpz_add_ui(generator, n, 1);  // generator = modulus + 1
    mpz_lcm(lambda, tmp1, tmp2);   // lamda = lcm(p-1, q-1)
    mpz_mul(n_square, n, n);
    mpz_t lambda_power;
    mpz_powm(lambda_power, generator, lambda, n_square);
    L_function(u, lambda_power, n);
    mpz_invert(u, u, N); // u = L((generator^lambda) mod n ^ 2) ) ^ -1 mod modulus
    mpz_clear(tmp1);
    mpz_clear(tmp2);
    mpz_clear(tmp3);
    mpz_clear(tmp4);
    mpz_clear(lambda_power);
}

void Paillier_GPU::encrypt(SyncArray<GHPair> &message){
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