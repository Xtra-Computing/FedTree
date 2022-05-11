#include "FedTree/Encryption/paillier_gpu.h"
#include "FedTree/util/device_lambda.cuh"
#include "gmp.h"
//#include "perf_tests/gpu_support.h"

void to_mpz(mpz_t r, uint32_t *x, uint32_t count) {
    mpz_import(r, count, -1, sizeof(uint32_t), 0, 0, x);
}

void from_mpz(mpz_t s, uint32_t *x, uint32_t count) {
    size_t words;

    if(mpz_sizeinbase(s, 2)>count*32) {
        fprintf(stderr, "from_mpz failed -- result does not fit\n");
        exit(1);
    }

    mpz_export(x, &words, -1, sizeof(uint32_t), 0, 0, s);
    while(words<count)
        x[words++]=0;
}

void cgbn_check(cgbn_error_report_t *report, const char *file=NULL, int32_t line=0) {
    // check for cgbn errors

    if(cgbn_error_report_check(report)) {
        printf("\n");
        printf("CGBN error occurred: %s\n", cgbn_error_string(report));

        if(report->_instance!=0xFFFFFFFF) {
            printf("Error reported by instance %d", report->_instance);
            if(report->_blockIdx.x!=0xFFFFFFFF || report->_threadIdx.x!=0xFFFFFFFF)
                printf(", ");
            if(report->_blockIdx.x!=0xFFFFFFFF)
                printf("blockIdx=(%d, %d, %d) ", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
            if(report->_threadIdx.x!=0xFFFFFFFF)
                printf("threadIdx=(%d, %d, %d)", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
            printf("\n");
        }
        else {
            printf("Error reported by blockIdx=(%d %d %d)", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
            printf("threadIdx=(%d %d %d)\n", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
        }
        if(file!=NULL)
            printf("file %s, line %d\n", file, line);
        exit(1);
    }
}

#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)

#define TPI 32
#define ENV_BITS 1024
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, ENV_BITS> env_t;

//template<uint32_t BITS>
void Paillier_GPU::add(mpz_t &result, mpz_t &x, mpz_t &y){
    mpz_mul(result, x, y);
    mpz_mod(result, result, paillier_cpu.n_square);
    return;
}

//template<uint32_t BITS>
void Paillier_GPU::mul(mpz_t result, mpz_t &x, mpz_t &y){
    mpz_powm(result, x, y, paillier_cpu.n_square);
    return ;
}

//template<uint32_t BITS>
void Paillier_GPU::L_function(mpz_t result, mpz_t input, mpz_t N){
    mpz_sub_ui(result, input, 1);
    mpz_tdiv_q(result, result, N);
}


void Paillier_GPU::parameters_cpu_to_gpu(){
    cgbn_mem_t<BITS> *n_cpu = (cgbn_mem_t<BITS> *)malloc(sizeof(cgbn_mem_t<BITS>));
    cgbn_mem_t<BITS> *n_square_cpu = (cgbn_mem_t<BITS> *)malloc(sizeof(cgbn_mem_t<BITS>));
    cgbn_mem_t<BITS> *generator_cpu = (cgbn_mem_t<BITS> *)malloc(sizeof(cgbn_mem_t<BITS>));
    cgbn_mem_t<BITS> *lambda_cpu = (cgbn_mem_t<BITS> *)malloc(sizeof(cgbn_mem_t<BITS>));
    cgbn_mem_t<BITS> *mu_cpu = (cgbn_mem_t<BITS> *)malloc(sizeof(cgbn_mem_t<BITS>));
    from_mpz(paillier_cpu.n, n_cpu->_limbs, BITS/32);
    from_mpz(paillier_cpu.n_square, n_square_cpu->_limbs, BITS/32);
    from_mpz(paillier_cpu.generator, generator_cpu->_limbs, BITS/32);
    from_mpz(paillier_cpu.lambda, lambda_cpu->_limbs, BITS/32);
    from_mpz(paillier_cpu.mu, mu_cpu->_limbs, BITS/32);

//    cgbn_mem_t<BITS> *n_gpu;
    CUDA_CHECK(cudaMalloc((void**)&n_gpu, sizeof(cgbn_mem_t<BITS>)));
    CUDA_CHECK(cudaMemcpy(n_gpu, n_cpu, sizeof(cgbn_mem_t<BITS>), cudaMemcpyHostToDevice));
//    cgbn_mem_t<BITS> *n_square_gpu;
    CUDA_CHECK(cudaMalloc((void**)&n_square_gpu, sizeof(cgbn_mem_t<BITS>)));
    CUDA_CHECK(cudaMemcpy(n_square_gpu, n_square_cpu, sizeof(cgbn_mem_t<BITS>), cudaMemcpyHostToDevice));
//    cgbn_mem_t<BITS> *generator_gpu;
    CUDA_CHECK(cudaMalloc((void**)&generator_gpu, sizeof(cgbn_mem_t<BITS>)));
    CUDA_CHECK(cudaMemcpy(generator_gpu, generator_cpu, sizeof(cgbn_mem_t<BITS>), cudaMemcpyHostToDevice));
//    cgbn_mem_t<BITS> *lambda_gpu;
    CUDA_CHECK(cudaMalloc((void**)&lambda_gpu, sizeof(cgbn_mem_t<BITS>)));
    CUDA_CHECK(cudaMemcpy(lambda_gpu, lambda_cpu, sizeof(cgbn_mem_t<BITS>), cudaMemcpyHostToDevice));
//    cgbn_mem_t<BITS> *mu_gpu;
    CUDA_CHECK(cudaMalloc((void**)&mu_gpu, sizeof(cgbn_mem_t<BITS>)));
    CUDA_CHECK(cudaMemcpy(mu_gpu, mu_cpu, sizeof(cgbn_mem_t<BITS>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    free(n_cpu);
    free(n_square_cpu);
    free(generator_cpu);
    free(lambda_cpu);
    free(mu_cpu);


//    cgbn_mem_t<BITS> *r_cpu = (cgbn_mem_t<BITS> *)malloc(sizeof(cgbn_mem_t<BITS>));
//    from_mpz(paillier_cpu.r, r_cpu->_limbs, BITS/32);
//    CUDA_CHECK(cudaMalloc((void**)&random_gpu, sizeof(cgbn_mem_t<BITS>)));
//    CUDA_CHECK(cudaMemcpy(random_gpu, r_cpu, sizeof(cgbn_mem_t<BITS>), cudaMemcpyHostToDevice));
//    free(r_cpu);
}
//template<uint32_t BITS>
void Paillier_GPU::keygen(){
    paillier_cpu.keyGen(BITS);
    parameters_cpu_to_gpu();

//    gmp_randstate_t state = new gmp_randstate_t();
//    gmp_randinit_mt(state);
////    gmp_randseed_ui(state, 1000U);
//    mpz tmp1, tmp2, tmp3, tmp4;
//    mpz_init(tmp1);
//    mpz_init(tmp2);
//    mpz_init(tmp3);
//    mpz_init(tmp4);
//    while (true){
//        mpz_urandomb(p, gpc_randstate, BITS/4);
//        mpz_urandomb(q, gpc_randstate, BITS/4);
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


}


__global__ void kernel_encrypt(cgbn_gh<BITS> *gh_gpu, cgbn_gh<BITS> *gh_results_gpu,
                               cgbn_mem_t<BITS> *generator_gpu, cgbn_mem_t<BITS> *random_gpu, cgbn_mem_t<BITS> *n_gpu,
                               cgbn_mem_t<BITS> *n_square_gpu, int n_instances){
    //        cgbm_mem_t<BITS> *g_gpu_test = new cgbm_mem_t<BITS>();
    int32_t idx;
    idx = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
    if(idx >= n_instances)
        return;

    context_t bn_context(cgbn_report_monitor);
//        context_t bn_context(cgbn_report_monitor);
    env_t bn_env(bn_context.env<env_t>());
    env_t::cgbn_t g, m, r, n, n_square, re1, re2, result, result2;
    env_t::cgbn_wide_t w;
    //todo: check whether g is in gpu or not. compare with another way: convert g to cgbn_mem_t before kernel
//        cgbn_set_ui32(bn_env, m, (uint32_t) (message_device_data[idx].g * 1e6));
    cgbn_load(bn_env, m, &(gh_gpu[idx].g));

//        cgbn_rem(bn_env, re1, m, m);

    cgbn_load(bn_env, g, generator_gpu);

//        cgbn_rem(bn_env, re1, g, g);
    cgbn_load(bn_env, r, random_gpu);
    cgbn_load(bn_env, n, n_gpu);
    cgbn_load(bn_env, n_square, n_square_gpu);

//        cgbn_rem(bn_env, g_mod, r, m);
//    cgbn_rem(bn_env, g_mod, g, g);

    // compute g_enc
    cgbn_modular_power(bn_env, re1, g, m, n_square);
    cgbn_modular_power(bn_env, re2, r, n, n_square);
    cgbn_mul_wide(bn_env, w, re1, re2);
    cgbn_rem_wide(bn_env, result, w, n_square);
    cgbn_store(bn_env, &(gh_results_gpu[idx].g), result);

//        // compute h_enc
////        cgbn_set_ui32(bn_env, m, (uint32_t) (message_device_data[idx].h * 1e6));
    cgbn_load(bn_env, m, &(gh_gpu[idx].h));
    cgbn_modular_power(bn_env, re1, g, m, n_square);
    cgbn_mul_wide(bn_env, w, re1, re2);
    cgbn_rem_wide(bn_env, result2, w, n_square);
    cgbn_store(bn_env, &(gh_results_gpu[idx].h), result2);
}

//template<uint32_t BITS>
void Paillier_GPU::encrypt(SyncArray<GHPair> &message){
//    auto message_device_data = message.device_data();

//    cgbn_error_report_t *report;
//    CUDA_CHECK(cgbn_error_report_alloc(&report));
    int n_instances = message.size();
    auto message_host_data = message.host_data();
    cgbn_gh<BITS> *gh_cpu = (cgbn_gh<BITS> *)malloc(sizeof(cgbn_gh<BITS>) * n_instances);
    mpz_t g_mpz, h_mpz;
    mpz_init(g_mpz);
    mpz_init(h_mpz);


//    std::cout<<"test import and export"<<std::endl;
//    float a = -0.2;
//    long a_ul = (long)(a * 1e6);
//    std::cout<<"import a_ul:"<<a_ul<<std::endl;
//    mpz_t tmp;
//    mpz_init(tmp);
//    mpz_import(tmp, 1, -1, sizeof(a_ul), 0, 0, &a_ul);
//    std::cout<<"import mpz:"<<tmp<<std::endl;
//    long a_l;
//    mpz_export(&a_l, (size_t*)0, -1, sizeof(a_l), 0, 0, tmp);
//    std::cout<<"export a_l:"<<a_l<<std::endl;
//    float a_res = (float) a_l/1e6;
//    std::cout<<"a_res:"<<a_res<<std::endl;



    for(int i = 0; i < n_instances; i++){
        long g_ul = (long) (message_host_data[i].g * 1e6);
//        if(i == 0)
//            std::cout<<"g_ul:"<<g_ul<<std::endl;
        long h_ul = (long) (message_host_data[i].h * 1e6);
        mpz_import(g_mpz, 1, -1, sizeof(g_ul), 0, 0, &g_ul);
//        if(i == 0)
//            std::cout<<"g_mpz:"<<g_mpz<<std::endl;
        mpz_import(h_mpz, 1, -1, sizeof(h_ul), 0, 0, &h_ul);
        from_mpz(g_mpz, gh_cpu[i].g._limbs, BITS / 32);
        from_mpz(h_mpz, gh_cpu[i].h._limbs, BITS / 32);
    }
    mpz_clear(g_mpz);
    mpz_clear(h_mpz);

    cgbn_gh<BITS> *gh_gpu;
    CUDA_CHECK(cudaMalloc((void**)&gh_gpu, sizeof(cgbn_gh<BITS>) * n_instances));
    CUDA_CHECK(cudaMemcpy(gh_gpu, gh_cpu, sizeof(cgbn_gh<BITS>) * n_instances, cudaMemcpyHostToDevice));
    free(gh_cpu);



    gmp_randstate_t state;
    gmp_randinit_mt(state);
//    gmp_randseed_ui(state, 1000U);
    mpz_t r;
    mpz_init(r);

    while(true) {
        mpz_urandomm(r, state, paillier_cpu.n);
        if(mpz_cmp_ui(r, 0))
            break;
    }
//    mpz_urandomb(r, state, BITS);
//    mpz_add_ui(r, r, 1); //ensure r > 0
//    mpz_mod(r, r, paillier_cpu.n);

    cgbn_mem_t<BITS> *random_cpu = (cgbn_mem_t<BITS> *)malloc(sizeof(cgbn_mem_t<BITS>));

    from_mpz(r, random_cpu->_limbs, BITS/32);
    cgbn_mem_t<BITS> *random_gpu;
    CUDA_CHECK(cudaMalloc((void**)&random_gpu, sizeof(cgbn_mem_t<BITS>)));
    CUDA_CHECK(cudaMemcpy(random_gpu, random_cpu, sizeof(cgbn_mem_t<BITS>), cudaMemcpyHostToDevice));
    free(random_cpu);
    mpz_clear(r);



    cgbn_gh<BITS>* gh_results_gpu;
    CUDA_CHECK(cudaMalloc((void **)&gh_results_gpu, sizeof(cgbn_gh<BITS>) * n_instances));


//    cudaMemcpy(&(gpuInstances->n), &n,, sizeof(n), cudaMemcpyHostToDevice);
    kernel_encrypt<<<(n_instances+3)/4, 128>>>(gh_gpu, gh_results_gpu, generator_gpu, random_gpu, n_gpu, n_square_gpu, n_instances);

    CUDA_CHECK(cudaDeviceSynchronize());
//    CGBN_CHECK(report);

//    CUDA_CHECK(cudaFree(gpu_parameters));
    cgbn_gh<BITS>* gh_results = (cgbn_gh<BITS>*)malloc(sizeof(cgbn_gh<BITS>)*n_instances);
    CUDA_CHECK(cudaMemcpy(gh_results, gh_results_gpu, sizeof(cgbn_gh<BITS>)*n_instances, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(gh_results_gpu));
//    auto message_host_data = message.host_data();
    for(int i = 0; i < n_instances; i++){
        mpz_init(message_host_data[i].g_enc);
        mpz_init(message_host_data[i].h_enc);
        // todo: another way: directly copy in GPU.
        to_mpz(message_host_data[i].g_enc, gh_results[i].g._limbs, BITS / 32);
        to_mpz(message_host_data[i].h_enc, gh_results[i].h._limbs, BITS / 32);
//        message_host_data[i].encrypted=true;
    }
    free(gh_results);
//    CUDA_CHECK(cgbn_error_report_free(report));
}

//template<uint32_t BITS>
//void Paillier_GPU::encrypt(GHPair &message){
////    cgbn_error_report_t *report;
////    CUDA_CHECK(cgbn_error_report_alloc(&report));
//
//
//    cgbn_gh<BITS> gh_cpu;
//    mpz_t g_mpz, h_mpz;
//    mpz_init(g_mpz);
//    mpz_init(h_mpz);
//    unsigned long g_ul = (unsigned long) (message.g * 1e6);
//    unsigned long h_ul = (unsigned long) (message.h * 1e6);
//    mpz_import(g_mpz, 1, -1, sizeof(g_ul), 0, 0, &g_ul);
//    mpz_import(h_mpz, 1, -1, sizeof(h_ul), 0, 0, &h_ul);
//    from_mpz(g_mpz, gh_cpu.g._limbs, BITS/32);
//    from_mpz(h_mpz, gh_cpu.h._limbs, BITS/32);
//    mpz_clear(g_mpz);
//    mpz_clear(h_mpz);
//
//    cgbn_gh<BITS> *gh_gpu;
//    CUDA_CHECK(cudaMalloc((void**)&gh_gpu, sizeof(cgbn_gh<BITS>)));
//    CUDA_CHECK(cudaMemcpy(gh_gpu, &gh_cpu, sizeof(cgbn_gh<BITS>), cudaMemcpyHostToDevice));
//
//    gmp_randstate_t state;
//    gmp_randinit_mt(state);
//    gmp_randseed_ui(state, 1000U);
//    mpz_t r;
//    mpz_init(r);
//    mpz_urandomb(r, state, BITS);
//    mpz_mod(r, r, paillier_cpu.n);
//
//    cgbn_mem_t<BITS> *random_cpu = (cgbn_mem_t<BITS> *)malloc(sizeof(cgbn_mem_t<BITS>));
//
//    from_mpz(r, random_cpu->_limbs, BITS/32);
//    cgbn_mem_t<BITS> *random_gpu;
//    CUDA_CHECK(cudaMalloc((void**)&random_gpu, sizeof(cgbn_mem_t<BITS>)));
//    CUDA_CHECK(cudaMemcpy(random_gpu, random_cpu, sizeof(cgbn_mem_t<BITS>), cudaMemcpyHostToDevice));
//
//    int n_instances = 1;
//    cgbn_gh<BITS>* gh_results_gpu;
//    CUDA_CHECK(cudaMalloc((void **)&gh_results_gpu, sizeof(cgbn_gh<BITS>) * n_instances));
//
////    cudaMemcpy(&(gpuInstances->n), &n,, sizeof(n), cudaMemcpyHostToDevice);
//
//    // todo: move values to gpu first
//    device_loop(1, [=] __device__(int idx){
////        context_t bn_context(cgbn_report_monitor, report, idx);
//        context_t bn_context(cgbn_report_monitor);
//        env_t bn_env(bn_context.env<env_t>());
//        env_t::cgbn_t g, m, r, n, n_square, re1, re2, result;
//        env_t::cgbn_wide_t w;
//
//        // todo: wrong! gh_cpu->g is cgbn_mem_t, in cpu, and cannot directly be multiplied with 1e6, check whether the computation is allowed
////        cgbn_set_ui32(bn_env, m, (uint32_t) (gh_cpu->g * 1e6));
//        cgbn_load(bn_env, m, &(gh_gpu->g));
//
//        cgbn_load(bn_env, g, generator_gpu);
//        cgbn_load(bn_env, r, random_gpu);
//        cgbn_load(bn_env, n, n_gpu);
//        cgbn_load(bn_env, n_square, n_square_gpu);
//
//        // compute g_enc
//        cgbn_modular_power(bn_env, re1, g, m, n_square);
//        cgbn_modular_power(bn_env, re2, r, n, n_square);
//        cgbn_mul_wide(bn_env, w, re1, re2);
//        cgbn_rem_wide(bn_env, result, w, n_square);
//        cgbn_store(bn_env, &(gh_results_gpu[idx].g), result);
//
//        // compute h_enc
////        cgbn_set_ui32(bn_env, m, (uint32_t) (gh_cpu.h * 1e6));
//        cgbn_load(bn_env, m, &(gh_gpu->h));
//        cgbn_modular_power(bn_env, re1, g, m, n_square);
//        cgbn_mul_wide(bn_env, w, re1, re2);
//        cgbn_rem_wide(bn_env, result, w, n_square);
//        cgbn_store(bn_env, &(gh_results_gpu[idx].h), result);
//    });
//
////    CGBN_CHECK(report);
//
////    CUDA_CHECK(cudaFree(gpu_parameters));
//    cgbn_gh<BITS>* gh_results = (cgbn_gh<BITS>*)malloc(sizeof(cgbn_gh<BITS>)*n_instances);
//    CUDA_CHECK(cudaMemcpy(gh_results, gh_results_gpu, sizeof(cgbn_gh<BITS>)*n_instances, cudaMemcpyDeviceToHost));
//    CUDA_CHECK(cudaFree(gh_results_gpu));
//
//    for(int i = 0; i < n_instances; i++){
//        mpz_init(message.g_enc);
//        mpz_init(message.h_enc);
//        // todo: another way: directly copy in GPU.
//        to_mpz(message.g_enc, gh_results->g._limbs, BITS / 32);
//        to_mpz(message.h_enc, gh_results->h._limbs, BITS / 32);
//    }
//    free(gh_results);
//}

__global__ void kernel_decrypt(cgbn_gh<BITS>* gh_enc_gpu, cgbn_mem_t<BITS> *lambda_gpu, cgbn_mem_t<BITS> *mu_gpu,
                               cgbn_mem_t<BITS> *n_gpu, cgbn_mem_t<BITS> *n_square_gpu, cgbn_gh<BITS> *gh_results_gpu,
                               int n_instances){
    int idx;
    idx = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
    if(idx >= n_instances)
        return;
    context_t bn_context(cgbn_report_monitor);
    env_t bn_env(bn_context.env<env_t>());
    env_t::cgbn_t c, lambda, n, mu, n_square, re1, re2, re3, re4, result, result2;

    cgbn_load(bn_env, c, &(gh_enc_gpu[idx].g));
    cgbn_load(bn_env, lambda, lambda_gpu);
    cgbn_load(bn_env, n, n_gpu);
    cgbn_load(bn_env, n_square, n_square_gpu);
    cgbn_load(bn_env, mu, mu_gpu);

    cgbn_modular_power(bn_env, re1, c, lambda, n_square);
    cgbn_sub_ui32(bn_env, re2, re1, 1);
    cgbn_div(bn_env, re3, re2, n);

    cgbn_mul(bn_env, re4, re3, mu);
    cgbn_rem(bn_env, result, re4, n);

    // todo: uint32 may not be enough to store g and h
//        message_device_data[idx].g = (float_type) cgbn_get_ui32(bn_env, result) / 1e6;
    cgbn_store(bn_env, &(gh_results_gpu[idx].g), result);
    // todo: check whether cpu mem data is syncrhonized or not

    cgbn_load(bn_env, c, &(gh_enc_gpu[idx].h));
    cgbn_modular_power(bn_env, re1, c, lambda, n_square);
    cgbn_sub_ui32(bn_env, re2, re1, 1);
    cgbn_div(bn_env, re3, re2, n);
    cgbn_mul(bn_env, re4, re3, mu);
    cgbn_rem(bn_env, result2, re4, n);
//        message_device_data[idx].h = (float_type) cgbn_get_ui32(bn_env, result) / 1e6;
    cgbn_store(bn_env, &(gh_results_gpu[idx].h), result2);
}
//template<uint32_t BITS>
void Paillier_GPU::decrypt(SyncArray<GHPair> &message){
//    auto message_device_data = message.device_data();
    auto message_host_data = message.host_data();
//    cgbn_error_report_t *report;
//    CUDA_CHECK(cgbn_error_report_alloc(&report));

    int n_instances = message.size();
    cgbn_gh<BITS>* gh_enc_cpu = (cgbn_gh<BITS>*)malloc(sizeof(cgbn_gh<BITS>)*n_instances);

    for(int i = 0; i < n_instances; i++){
        if(message_host_data[i].encrypted) {
            from_mpz(message_host_data[i].g_enc, gh_enc_cpu[i].g._limbs, BITS / 32);
            from_mpz(message_host_data[i].h_enc, gh_enc_cpu[i].h._limbs, BITS / 32);
        }
    }
    cgbn_gh<BITS>* gh_enc_gpu;
    CUDA_CHECK(cudaMalloc((void **)&gh_enc_gpu, sizeof(cgbn_gh<BITS>) * n_instances));
    CUDA_CHECK(cudaMemcpy(gh_enc_gpu, gh_enc_cpu, sizeof(cgbn_gh<BITS>) * n_instances, cudaMemcpyHostToDevice));


    cgbn_gh<BITS>* gh_results_gpu;
    CUDA_CHECK(cudaMalloc((void **)&gh_results_gpu, sizeof(cgbn_gh<BITS>) * n_instances));
//    std::cout<<"n_instances:"<<n_instances<<std::endl;
    kernel_decrypt<<<(n_instances+3)/4, 128>>>(gh_enc_gpu, lambda_gpu, mu_gpu, n_gpu, n_square_gpu, gh_results_gpu, n_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    cgbn_gh<BITS>* gh_results = (cgbn_gh<BITS>*)malloc(sizeof(cgbn_gh<BITS>)*n_instances);
    CUDA_CHECK(cudaMemcpy(gh_results, gh_results_gpu, sizeof(cgbn_gh<BITS>)*n_instances, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(gh_results_gpu));
//    auto message_host_data = message.host_data();
    mpz_t g_result, h_result;
    mpz_init(g_result);
    mpz_init(h_result);
    for(int i = 0; i < n_instances; i++){
        if(message_host_data[i].encrypted) {
            long g_ul = 0, h_ul = 0;
            to_mpz(g_result, gh_results[i].g._limbs, BITS / 32);
            to_mpz(h_result, gh_results[i].h._limbs, BITS / 32);
            mpz_export(&g_ul, 0, -1, sizeof(g_ul), 0, 0, g_result);
            mpz_export(&h_ul, 0, -1, sizeof(h_ul), 0, 0, h_result);
            message_host_data[i].g = (float_type) g_ul / 1e6;
            message_host_data[i].h = (float_type) h_ul / 1e6;
        }
    }
    free(gh_results);
    mpz_clear(g_result);
    mpz_clear(h_result);
}


void Paillier_GPU::decrypt(GHPair &message){
//    auto message_device_data = message.device_data();
    auto message_host_data = message;
//    cgbn_error_report_t *report;
//    CUDA_CHECK(cgbn_error_report_alloc(&report));

    int n_instances = 1;
    cgbn_gh<BITS>* gh_enc_cpu = (cgbn_gh<BITS>*)malloc(sizeof(cgbn_gh<BITS>)*n_instances);

    if(message.encrypted) {
        from_mpz(message.g_enc, gh_enc_cpu[0].g._limbs, BITS / 32);
        from_mpz(message.h_enc, gh_enc_cpu[0].h._limbs, BITS / 32);
    }

    cgbn_gh<BITS>* gh_enc_gpu;
    CUDA_CHECK(cudaMalloc((void **)&gh_enc_gpu, sizeof(cgbn_gh<BITS>) * n_instances));
    CUDA_CHECK(cudaMemcpy(gh_enc_gpu, gh_enc_cpu, sizeof(cgbn_gh<BITS>) * n_instances, cudaMemcpyHostToDevice));


    cgbn_gh<BITS>* gh_results_gpu;
    CUDA_CHECK(cudaMalloc((void **)&gh_results_gpu, sizeof(cgbn_gh<BITS>) * n_instances));
//    std::cout<<"n_instances:"<<n_instances<<std::endl;
    kernel_decrypt<<<(n_instances+3)/4, 128>>>(gh_enc_gpu, lambda_gpu, mu_gpu, n_gpu, n_square_gpu, gh_results_gpu, n_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    cgbn_gh<BITS>* gh_results = (cgbn_gh<BITS>*)malloc(sizeof(cgbn_gh<BITS>)*n_instances);
    CUDA_CHECK(cudaMemcpy(gh_results, gh_results_gpu, sizeof(cgbn_gh<BITS>)*n_instances, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(gh_results_gpu));
//    auto message_host_data = message.host_data();
    mpz_t g_result, h_result;
    mpz_init(g_result);
    mpz_init(h_result);

    if(message.encrypted) {
        long g_ul = 0, h_ul = 0;
        to_mpz(g_result, gh_results[0].g._limbs, BITS / 32);
        to_mpz(h_result, gh_results[0].h._limbs, BITS / 32);
        mpz_export(&g_ul, 0, -1, sizeof(g_ul), 0, 0, g_result);
        mpz_export(&h_ul, 0, -1, sizeof(h_ul), 0, 0, h_result);
        message.g = (float_type) g_ul / 1e6;
        message.h = (float_type) h_ul / 1e6;
    }

    free(gh_results);
    mpz_clear(g_result);
    mpz_clear(h_result);
}

//template<uint32_t BITS>
//void Paillier_GPU::decrypt(GHPair &message){
////    auto message_device_data = message.device_data();
////    cgbn_error_report_t *report;
////    CUDA_CHECK(cgbn_error_report_alloc(&report));
//
//    cgbn_gh<BITS>* gh_enc_cpu = (cgbn_gh<BITS>*)malloc(sizeof(cgbn_gh<BITS>));
//    from_mpz(message.g_enc, gh_enc_cpu->g._limbs, BITS / 32);
//    from_mpz(message.h_enc, gh_enc_cpu->h._limbs, BITS / 32);
//    cgbn_gh<BITS>* gh_enc_gpu;
//    CUDA_CHECK(cudaMalloc((void **)&gh_enc_gpu, sizeof(cgbn_gh<BITS>)));
//    CUDA_CHECK(cudaMemcpy(gh_enc_gpu, gh_enc_cpu, sizeof(cgbn_gh<BITS>), cudaMemcpyHostToDevice));
//
//    int n_instances = 1;
//    cgbn_gh<BITS>* gh_results_gpu;
//    CUDA_CHECK(cudaMalloc((void **)&gh_results_gpu, sizeof(cgbn_gh<BITS>) * n_instances));
//
//
////    float_type* g_gpu, h_gpu;
////    CUDA_CHECK(cudaMalloc((void **)&g_gpu, sizeof(float_type)));
////    CUDA_CHECK(cudaMalloc((void **)&h_gpu, sizeof(float_type)));
//
//    device_loop(1, [=] __device__(int idx){
////        context_t bn_context(cgbn_report_monitor, report, idx);
//        context_t bn_context(cgbn_report_monitor);
//        env_t bn_env(bn_context.env<env_t>());
//        env_t::cgbn_t c, lambda, n, mu, n_square, re1, re2, re3, re4, result;
//
//        cgbn_load(bn_env, c, &(gh_enc_gpu->g));
//        cgbn_load(bn_env, lambda, this->lambda_gpu);
//        cgbn_load(bn_env, n, this->n_gpu);
//        cgbn_load(bn_env, n_square, this->n_square_gpu);
//
//        cgbn_modular_power(bn_env, re1, c, lambda, n_square);
//        cgbn_sub_ui32(bn_env, re2, re1, 1);
//        cgbn_div(bn_env, re3, re2, n);
//
//        cgbn_mul(bn_env, re4, re3, mu);
//        cgbn_rem(bn_env, result, re4, n);
//
////        *g_gpu = (float_type) cgbn_get_ui32(bn_env, result) / 1e6;
//        cgbn_store(bn_env, &(gh_results_gpu[idx].g), result);
//        // todo: check whether cpu mem data is syncrhonized or not
//
//        cgbn_load(bn_env, c, &(gh_enc_gpu->h));
//        cgbn_modular_power(bn_env, re1, c, lambda, n_square);
//        cgbn_sub_ui32(bn_env, re2, re1, 1);
//        cgbn_div(bn_env, re3, re2, n);
//        cgbn_mul(bn_env, re4, re3, mu);
//        cgbn_rem(bn_env, result, re4, n);
////        *h_gpu = (float_type) cgbn_get_ui32(bn_env, result) / 1e6;
//        cgbn_store(bn_env, &(gh_results_gpu[idx].h), result);
//    });
//
//    cgbn_gh<BITS>* gh_results = (cgbn_gh<BITS>*)malloc(sizeof(cgbn_gh<BITS>)*n_instances);
//    CUDA_CHECK(cudaMemcpy(gh_results, gh_results_gpu, sizeof(cgbn_gh<BITS>)*n_instances, cudaMemcpyDeviceToHost));
//    CUDA_CHECK(cudaFree(gh_results_gpu));
//
//    mpz_t g_result, h_result;
//    mpz_init(g_result);
//    mpz_init(h_result);
//    unsigned long g_ul, h_ul;
//
//    to_mpz(g_result, gh_results->g._limbs, BITS/32);
//    to_mpz(h_result, gh_results->h._limbs, BITS/32);
//    mpz_export(&g_ul, 0, -1, sizeof(g_ul), 0, 0, g_result);
//    mpz_export(&h_ul, 0, -1, sizeof(h_ul), 0, 0, h_result);
//    message.g = (float_type) g_ul/1e6;
//    message.h = (float_type) h_ul/1e6;
//
//    free(gh_results);
//    mpz_clear(g_result);
//    mpz_clear(h_result);
////    CUDA_CHECK(cudaMemcpy(message.g, g_gpu, sizeof(float_type), cudaMemcpyDeviceToHost));
////    CUDA_CHECK(cudaMemcpy(message.h, h_gpu, sizeof(float_type), cudaMemcpyDeviceToHost));
//}