#include "FedTree/Encryption/paillier_gpu.cuh"

template<uint32_t BITS>
void Paillier_GPU<BITS>::add(mpz_t &result, mpz_t &x, mpz_t &y){
    mpz_mul(result, x, y);
    mpz_mod(result, result, paillier_cpu.n_square);
    return;
}

template<uint32_t BITS>
void Paillier_GPU<BITS>::mul(mpz_t result, mpz_t &x, mpz_t &y){
    mpz_powm(result, x, y, paillier_cpu.n_square);
    return ;
}

template<uint32_t BITS>
void Paillier_GPU<BITS>::L_function(mpz_t result, mpz_t input, mpz_t N){
    mpz_sub_ui(result, input, 1);
    mpz_tdiv_q(result, result, N);
}

template<uint32_t BITS>
void Paillier_GPU<BITS>::keygen(){
    paillier_cpu.keygen(this->key_length);

//    gmp_randstate_t state = new gmp_randstate_t();
//    gmp_randinit_mt(state);
////    gmp_randseed_ui(state, 1000U);
//    mpz tmp1, tmp2, tmp3, tmp4;
//    mpz_init(tmp1);
//    mpz_init(tmp2);
//    mpz_init(tmp3);
//    mpz_init(tmp4);
//    while (true){
//        mpz_urandomb(p, gpc_randstate, key_length/4);
//        mpz_urandomb(q, gpc_randstate, key_length/4);
//        mpz_nextprime(p, p);
//        mpz_nextprime(q, q);
//        mpz_sub_ui(tmp1, p, 1);
//        mpz_sub_ui(tmp2, q, 1);
//        mpz_mul(tmp3, tmp1, tmp2); // tmp3 = (p-1)(q-1)
//        mpz_mul(tmp4, p, q); // tmp4 = p*q
//        mpz_gcd(tmp3, tmp3, tmp4); // tmp = gcd(pq, (p-1)(q-1))
//        if(mpz_cmp_ui(tmp3, 1) == 0) // gcd(pq, (p-1)(q-1)) == 1
//            break;
//    }
//
//    n = tmp4;                                                        // n = p * q
//    mpz_add_ui(generator, n, 1);  // generator = modulus + 1
//    mpz_lcm(lambda, tmp1, tmp2);   // lamda = lcm(p-1, q-1)
//    mpz_mul(n_square, n, n);
//    mpz_t lambda_power;
//    mpz_init(lambda_power);
//    mpz_powm(lambda_power, generator, lambda, n_square);
//    L_function(mu, lambda_power, n);
//    mpz_invert(mu, mu, N); // u = L((generator^lambda) mod n ^ 2) ) ^ -1 mod modulus
//    mpz_clear(tmp1);
//    mpz_clear(tmp2);
//    mpz_clear(tmp3);
//    mpz_clear(tmp4);
//    mpz_clear(lambda_power);


    cgbn_mem_t<key_length> *n_cpu = (cgbn_mem_t<key_length> *)malloc(sizeof(cgbn_mem_t<key_length>));
    cgbn_mem_t<key_length> *n_square_cpu = (cgbn_mem_t<key_length> *)malloc(sizeof(cgbn_mem_t<key_length>));
    cgbn_mem_t<key_length> *generator_cpu = (cgbn_mem_t<key_length> *)malloc(sizeof(cgbn_mem_t<key_length>));
    cgbn_mem_t<key_length> *lambda_cpu = (cgbn_mem_t<key_length> *)malloc(sizeof(cgbn_mem_t<key_length>));
    cgbn_mem_t<key_length> *mu_cpu = (cgbn_mem_t<key_length> *)malloc(sizeof(cgbn_mem_t<key_length>));
    from_mpz(paillier_cpu.n, n_cpu._limbs, key_length/32);
    from_mpz(paillier_cpu.n_square, n_square_cpu._limbs, key_length/32);
    from_mpz(paillier_cpu.generator, generator_cpu._limbs, key_length/32);
    from_mpz(paillier_cpu.lambda, lambda_cpu._limbs, key_length/32);
    from_mpz(paillier_cpu.mu, mu_cpu._limbs, key_length/32);

    cgbn_mem_t<key_length> *n_gpu;
    CUDA_CHECK(cudaMalloc((void**)&n_gpu, sizeof(cgbn_mem_t<key_length>)));
    CUDA_CHECK(cudaMemcpy(n_gpu, n_cpu, sizeof(cgbn_mem_t<key_length>), cudaMemcpyHostToDevice));
    cgbn_mem_t<key_length> *n_square_gpu;
    CUDA_CHECK(cudaMalloc((void**)&n_square_gpu, sizeof(cgbn_mem_t<key_length>)));
    CUDA_CHECK(cudaMemcpy(n_square_gpu, n_square_cpu, sizeof(cgbn_mem_t<key_length>), cudaMemcpyHostToDevice));
    cgbn_mem_t<key_length> *generator_gpu;
    CUDA_CHECK(cudaMalloc((void**)&generator_gpu, sizeof(cgbn_mem_t<key_length>)));
    CUDA_CHECK(cudaMemcpy(generator_gpu, generator_cpu, sizeof(cgbn_mem_t<key_length>), cudaMemcpyHostToDevice));
    cgbn_mem_t<key_length> *lambda_gpu;
    CUDA_CHECK(cudaMalloc((void**)&lambda_gpu, sizeof(cgbn_mem_t<key_length>)));
    CUDA_CHECK(cudaMemcpy(lambda_gpu, lambda_cpu, sizeof(cgbn_mem_t<key_length>), cudaMemcpyHostToDevice));
    cgbn_mem_t<key_length> *mu_gpu;
    CUDA_CHECK(cudaMalloc((void**)&mu_gpu, sizeof(cgbn_mem_t<key_length>)));
    CUDA_CHECK(cudaMemcpy(mu_gpu, mu_cpu, sizeof(cgbn_mem_t<key_length>), cudaMemcpyHostToDevice));

    free(n_cpu);
    free(n_square_cpu);
    free(generator_cpu);
    free(lambda_cpu);
    free(mu_cpu);
}

template<uint32_t BITS>
void Paillier_GPU<BITS>::encrypt(SyncArray<GHPair> &message){
    auto message_device_data = message.device_data();
    cgbn_error_report_t *report;
    CUDA_CHECK(cgbn_error_report_alloc(&report));


//    cgbn_pailler_encryption_parameters<key_length> *gpu_parameters;
//    cgbn_pailler_encryption_parameters<key_length> *parameters = (cgbn_pailler_encryption_parameters<key_length> *)
//            malloc(sizeof(cgbn_pailler_encryption_parameters<key_length>));
//    CUDA_CHECK(cudaMalloc((void **)&gpu_parameters, sizeof(cgbn_pailler_encryption_parameters<key_length>)));
//    //todo: n is an array? check whether it is correct
//    cgbn_mem_t<key_length> n_cgbn, n_square_cgbn, g_cgbn;
//    from_mpz(n, parameters->n_cgbn._limbs, key_length/32);
//    from_mpz(n_square, parameters->n_square_cgbn._limbs, key_length/32);
//    from_mpz(generator, parameters->g_cgbn._limbs, key_length/32);

//    gmp_randstate_t state = new gmp_randstate_t();
//    gmp_randinit_mt(state);
//    gmp_randseed_ui(state, 1000U);
//    mpz_t r;
//    mpz_init(r);
//    mpz_urandomm(r, state, key_length);
//    mpz_mod(r, r, modulus_cpu);
//    from_mpz(r, gpu_parameters->random, key_length/32);

//
//    CUDA_CHECK(cudaMemcpy(gpu_parameters, parameters, sizeof(cgbn_pailler_encryption_parameters<key_length>), cudaMemcpyHostToDevice));
//    free(parameters);


    gmp_randstate_t state = new gmp_randstate_t();
    gmp_randinit_mt(state);
    gmp_randseed_ui(state, 1000U);
    mpz_t r;
    mpz_init(r);
    mpz_urandomm(r, state, key_length);
    mpz_mod(r, r, modulus_cpu);

    cgbn_mem_t<key_length> *random_cpu = (cgbn_mem_t<key_length> *)malloc(sizeof(cgbn_mem_t<key_length>));

    from_mpz(r, random_cpu, key_length/32);
    cgbn_mem_t<key_length> *random_gpu;
    CUDA_CHECK(cudaMalloc((void**)&random_gpu, sizeof(cgbn_mem_t<key_length>)));
    CUDA_CHECK(cudaMemcpy(random_gpu, random_cpu, sizeof(cgbn_mem_t<key_length>), cudaMemcpyHostToDevice));


    int n_instances = message.size();
    cgbn_gh_results<key_length>* gh_results_gpu;
    CUDA_CHECK(cudaMalloc((void **)&gh_results_gpu, sizeof(cgbn_gh_results<key_length>) * n_instances));

//    cudaMemcpy(&(gpuInstances->n), &n,, sizeof(n), cudaMemcpyHostToDevice);

    // todo: move values to gpu first
    device_loop(n_instances, [=] __device__(int idx){
        context_t bn_context(cgbn_report_monitor, report, idx);
        env_t bn_env(bn_context.env<env_t>());
        env_t::cgbn_t g, m, r, n, n_square, re1, re2, result;
        env_t::cgbn_wide_t w;
        //todo: check whether g is in gpu or not. compare with another way: convert g to cgbn_mem_t before kernel
        cgbn_set_ui32(bn_env, m, (uint32_t) (message_device_data[idx].g * 1e6));

        cgbn_load(bn_env, g, generator_gpu);
        cgbn_load(bn_env, r, random_gpu);
        cgbn_load(bn_env, n, n_gpu);
        cgbn_load(bn_env, n_square, n_square_gpu);

        // compute g_enc
        cgbn_modular_power(bn_env, re1, g, m, n_square);
        cgbn_modular_power(bn_env, re2, r, n, n_square);
        cgbn_mul_wide(bn_env, w, re1, re2);
        cgbn_rem_wide(bn_env, result, w, n_square);
        cgbn_store(bn_env, &(gh_results_gpu[idx].g), result);

        // compute h_enc
        cgbn_set_ui32(bn_env, m, (uint32_t) (message_device_data[idx].h * 1e6));
        cgbn_modular_power(bn_env, re1, g, m, n_square);
        cgbn_mul_wide(bn_env, w, re1, re2);
        cgbn_rem_wide(bn_env, result, w, n_square);
        cgbn_store(bn_env, &(gh_results_gpu[idx].h), result);
    });

    CGBN_CHECK(report);

//    CUDA_CHECK(cudaFree(gpu_parameters));
    cgbn_gh_results<key_length>* gh_results = (cgbn_gh_results<key_length>*)malloc(sizeof(cgbn_gh_results<key_length>)*n_instances);
    CUDA_CHECK(cudaMemcpy(gh_results, gh_results_gpu, sizeof(cgbn_gh_results<key_length>)*n_instances, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(gh_results_gpu));
    auto message_host_data = message.host_data();
    for(int i = 0; i < n_instances; i++){
        mpz_init(message_host_data[i].g_enc);
        mpz_init(message_host_data[i].h_enc);
        // todo: another way: directly copy in GPU.
        to_mpz(message_host_data[i].g_enc, gh_results.g._impl, key_length / 32);
        to_mpz(message_host_data[i].h_enc, gh_results.h._impl, key_length / 32);
    }
    free(gh_results);
}

template<uint32_t BITS>
void Paillier_GPU<BITS>::encrypt(GHPair &message){
//    auto message_device_data = message.device_data();

    cgbn_error_report_t *report;
    CUDA_CHECK(cgbn_error_report_alloc(&report));

    cgbn_gh_results<this->key_length>* gh_cpu = (cgbn_gh_results<key_length>*)malloc(sizeof(cgbn_gh_results<key_length>));
    from_mpz(message.g, gh_cpu.g._impl, key_length / 32);
    from_mpz(message.h, gh_cpu.h._impl, key_length / 32);

    //    cgbn_pailler_encryption_parameters<key_length> *gpu_parameters;
//    cgbn_pailler_encryption_parameters<key_length> *parameters = (cgbn_pailler_encryption_parameters<key_length> *)
//            malloc(sizeof(cgbn_pailler_encryption_parameters<key_length>));
//    CUDA_CHECK(cudaMalloc((void **)&gpu_parameters, sizeof(cgbn_pailler_encryption_parameters<key_length>)));
//    //todo: n is an array? check whether it is correct
//    cgbn_mem_t<key_length> n_cgbn, n_square_cgbn, g_cgbn;
//    from_mpz(n, parameters->n_cgbn._limbs, key_length/32);
//    from_mpz(n_square, parameters->n_square_cgbn._limbs, key_length/32);
//    from_mpz(generator, parameters->g_cgbn._limbs, key_length/32);

//    gmp_randstate_t state = new gmp_randstate_t();
//    gmp_randinit_mt(state);
//    gmp_randseed_ui(state, 1000U);
//    mpz_t r;
//    mpz_init(r);
//    mpz_urandomm(r, state, key_length);
//    mpz_mod(r, r, modulus_cpu);
//    from_mpz(r, gpu_parameters->random, key_length/32);

//
//    CUDA_CHECK(cudaMemcpy(gpu_parameters, parameters, sizeof(cgbn_pailler_encryption_parameters<key_length>), cudaMemcpyHostToDevice));
//    free(parameters);


    gmp_randstate_t state = new gmp_randstate_t();
    gmp_randinit_mt(state);
    gmp_randseed_ui(state, 1000U);
    mpz_t r;
    mpz_init(r);
    mpz_urandomm(r, state, key_length);
    mpz_mod(r, r, modulus_cpu);

    cgbn_mem_t<key_length> *random_cpu = (cgbn_mem_t<key_length> *)malloc(sizeof(cgbn_mem_t<key_length>));

    from_mpz(r, random_cpu, key_length/32);
    cgbn_mem_t<key_length> *random_gpu;
    CUDA_CHECK(cudaMalloc((void**)&random_gpu, sizeof(cgbn_mem_t<key_length>)));
    CUDA_CHECK(cudaMemcpy(random_gpu, random_cpu, sizeof(cgbn_mem_t<key_length>), cudaMemcpyHostToDevice));


    int n_instances = 1;
    cgbn_gh_results<key_length>* gh_results_gpu;
    CUDA_CHECK(cudaMalloc((void **)&gh_results_gpu, sizeof(cgbn_gh_results<key_length>) * n_instances));

//    cudaMemcpy(&(gpuInstances->n), &n,, sizeof(n), cudaMemcpyHostToDevice);

    // todo: move values to gpu first
    device_loop(1, [=] __device__(int idx){
        context_t bn_context(cgbn_report_monitor, report, idx);
        env_t bn_env(bn_context.env<env_t>());
        env_t::cgbn_t g, m, r, n, n_square, re1, re2, result;
        env_t::cgbn_wide_t w;
        //todo: check whether g is in gpu or not. compare with another way: convert g to cgbn_mem_t before kernel
        cgbn_set_ui32(bn_env, m, (uint32_t) (gh_cpu.g * 1e6));

        cgbn_load(bn_env, g, generator_gpu);
        cgbn_load(bn_env, r, random_gpu);
        cgbn_load(bn_env, n, n_gpu);
        cgbn_load(bn_env, n_square, n_square_gpu);

        // compute g_enc
        cgbn_modular_power(bn_env, re1, g, m, n_square);
        cgbn_modular_power(bn_env, re2, r, n, n_square);
        cgbn_mul_wide(bn_env, w, re1, re2);
        cgbn_rem_wide(bn_env, result, w, n_square);
        cgbn_store(bn_env, &(gh_results_gpu[idx].g), result);

        // compute h_enc
        cgbn_set_ui32(bn_env, m, (uint32_t) (gh_cpu.h * 1e6));
        cgbn_modular_power(bn_env, re1, g, m, n_square);
        cgbn_mul_wide(bn_env, w, re1, re2);
        cgbn_rem_wide(bn_env, result, w, n_square);
        cgbn_store(bn_env, &(gh_results_gpu[idx].h), result);
    });

    CGBN_CHECK(report);

//    CUDA_CHECK(cudaFree(gpu_parameters));
    cgbn_gh_results<key_length>* gh_results = (cgbn_gh_results<key_length>*)malloc(sizeof(cgbn_gh_results<key_length>)*n_instances);
    CUDA_CHECK(cudaMemcpy(gh_results, gh_results_gpu, sizeof(cgbn_gh_results<key_length>)*n_instances, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(gh_results_gpu));

    for(int i = 0; i < n_instances; i++){
        mpz_init(message.g_enc);
        mpz_init(message.h_enc);
        // todo: another way: directly copy in GPU.
        to_mpz(message.g_enc, gh_results.g._impl, key_length / 32);
        to_mpz(message.h_enc, gh_results.h._impl, key_length / 32);
    }
    free(gh_results);
}
template<uint32_t BITS>
void Paillier_GPU<BITS>::decrypt(SyncArray<GHPair> &message){
//    auto message_device_data = message.device_data();
    auto message_host_data = message.host_data();
    cgbn_error_report_t *report;
    CUDA_CHECK(cgbn_error_report_alloc(&report));

    int n_instances = message.size();
    cgbn_gh_results<this->key_length>* gh_enc_cpu = (cgbn_gh_results<key_length>*)malloc(sizeof(cgbn_gh_results<key_length>)*n_instances);
    for(int i = 0; i < n_instances; i++){
        from_mpz(message_host_data[i].g_enc, gh_enc_cpu->g._impl, key_length / 32);
        from_mpz(message_host_data[i].h_enc, gh_enc_cpu->h._impl, key_length / 32);
    }
    cgbn_gh_results<key_length>* gh_enc_gpu;
    CUDA_CHECK(cudaMalloc((void **)&gh_enc_gpu, sizeof(cgbn_gh_results<key_length>) * n_instances));
    CUDA_CHECK(cudaMemcpy(gh_enc_gpu, gh_enc_cpu, sizeof(cgbn_gh_results<key_length>) * n_instances, cudaMemcpyHostToDevice));

    device_loop(n_instances, [=] __device__(int idx){
        context_t bn_context(cgbn_report_monitor, report, idx);
        env_t bn_env(bn_context.env<env_t>());
        env_t::cgbn_t c, lambda, n, mu, n_square, re1, re2, re3, re4;

        cgbn_load(bn_env, c, &gh_enc_gpu[idx].g);
        cgbn_load(bn_env, lambda, &this->lambda);
        cgbn_load(bn_env, n, &this->n);
        cgbn_sqr(bn_env, n_square, &this->n_square);

        cgbn_modular_power(bn_env, re1, c, lambda, n_square);
        cgbn_sub_ui32(bn_env, re2, re1, 1);
        cgbn_div(bn_env, re3, re2, n);

        cgbn_mul(bn_env, re4, re3, mu);
        cgbn_rem(bn_env, result, re4, n);

        message_device_data[idx].g = (float_type) cgbn_get_ui32(bn_env, result) / 1e6;
        // todo: check whether cpu mem data is syncrhonized or not

        cgbn_load(bn_env, c, &gh_enc_gpu[idx].h);
        cgbn_modular_power(bn_env, re1, c, lambda, n_square);
        cgbn_sub_ui32(bn_env, re2, re1, 1);
        cgbn_div(bn_env, re3, re2, n);
        cgbn_mul(bn_env, re4, re3, mu);
        cgbn_rem(bn_env, result, re4, n);
        message_device_data[idx].h = (float_type) cgbn_get_ui32(bn_env, result) / 1e6;
    });
}


template<uint32_t BITS>
void Paillier_GPU<BITS>::decrypt(GHPair &message){
//    auto message_device_data = message.device_data();
    cgbn_error_report_t *report;
    CUDA_CHECK(cgbn_error_report_alloc(&report));

    cgbn_gh_results<key_length>* gh_enc_cpu = (cgbn_gh_results<key_length>*)malloc(sizeof(cgbn_gh_results<key_length>));
    from_mpz(message.g_enc, gh_enc_cpu.g._impl, key_length / 32);
    from_mpz(message.h_enc, gh_enc_cpu.h._impl, key_length / 32);
    cgbn_gh_results<key_length>* gh_enc_gpu;
    CUDA_CHECK(cudaMalloc((void **)&gh_enc_gpu, sizeof(cgbn_gh_results<key_length>)));
    CUDA_CHECK(cudaMemcpy(gh_enc_gpu, gh_enc_cpu, sizeof(cgbn_gh_results<key_length>), cudaMemcpyHostToDevice));

    float_type* g_gpu, h_gpu;
    CUDA_CHECK(cudaMalloc((void **)&g_gpu, sizeof(float_type)));
    CUDA_CHECK(cudaMalloc((void **)&h_gpu, sizeof(float_type)));
    device_loop(1, [=] __device__(int idx){
        context_t bn_context(cgbn_report_monitor, report, idx);
        env_t bn_env(bn_context.env<env_t>());
        env_t::cgbn_t c, lambda, n, mu, n_square, re1, re2, re3, re4;

        cgbn_load(bn_env, c, &gh_enc_gpu.g);
        cgbn_load(bn_env, lambda, &this->lambda);
        cgbn_load(bn_env, n, &this->n);
        cgbn_sqr(bn_env, n_square, &this->n_square);

        cgbn_modular_power(bn_env, re1, c, lambda, n_square);
        cgbn_sub_ui32(bn_env, re2, re1, 1);
        cgbn_div(bn_env, re3, re2, n);

        cgbn_mul(bn_env, re4, re3, mu);
        cgbn_rem(bn_env, result, re4, n);

        *g_gpu = (float_type) cgbn_get_ui32(bn_env, result) / 1e6;
        // todo: check whether cpu mem data is syncrhonized or not

        cgbn_load(bn_env, c, &gh_enc_gpu.h);
        cgbn_modular_power(bn_env, re1, c, lambda, n_square);
        cgbn_sub_ui32(bn_env, re2, re1, 1);
        cgbn_div(bn_env, re3, re2, n);
        cgbn_mul(bn_env, re4, re3, mu);
        cgbn_rem(bn_env, result, re4, n);
        *h_gpu = (float_type) cgbn_get_ui32(bn_env, result) / 1e6;
    });
    CUDA_CHECK(cudaMemcpy(message.g, g_gpu, sizeof(float_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(message.h, h_gpu, sizeof(float_type), cudaMemcpyDeviceToHost));
}