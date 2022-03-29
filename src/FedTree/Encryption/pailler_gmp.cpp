#include "FedTree/Encryption/paillier_gmp.h"

void Paillier_GMP::Paillier_GMP(){
    mpz_init(n);
    mpz_init(n_square);
    mpz_init(generator);
    mpz_init(p);
    mpz_init(q);
    mpz_init(lambda);
    mpz_init(mu);
}
void Paillier_GMP::add(mpz_t &result, mpz_t &x, mpz_t &y){
    mpz_mul(result, x, y);
    mpz_mod(result, result, n_square);
    return;
}

void Paillier_GMP::mul(mpz_t &result, mpz_t &x, mpz_t &y){
    mpz_powm(result, x, y, n_square);
    return ;
}

void Paillier_GMP::L_function(mpz_t &result, mpz_t &input, mpz_t &N){
    mpz_sub_ui(result, input, 1);
    mpz_tdiv_q(result, result, N);
    return;
}

void Paillier_GMP::encrypt(mpz_t &result, mpz_t &message){
    gmp_randstate_t state = new gmp_randstate_t();
    gmp_randinit_mt(state);
    gmp_randseed_ui(state, 1000U);
    mpz_t r;
    mpz_init(r);
    mpz_urandomm(r, state, key_length);
    mpz_mod(r, r, modulus_cpu);
    mpz_powm(result, r, n, n_square);
    //since g=1+n, g^m=(1+n)^m=1+nm % n^2
    mpz_mul(r, message, n);
    mpz_add_ui(r, r, 1);
    mpz_mul(result, result, r);
    mpz_mod(result, result, n_square);
    mpz_clear(r);
    return;
}

void Paillier_GMP::decrypt(mpz_t &result, mpz_t &message){
    mpz_powm(result, message, lambda, n_square);
    L_function(result, result, n);
    mpz_mul(result, result, mu);
    mpz_mod(result, result, n);
    return;
}

void Paillier_GMP::keyGen(uint32_t keyLength) {
    this->key_length = keyLength;

    gmp_randstate_t state = new gmp_randstate_t();
    gmp_randinit_mt(state);
//    gmp_randseed_ui(state, 1000U);
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
    mpz_add_ui(generator, n, 1);  // g = n + 1
    mpz_lcm(lambda, tmp1, tmp2);   // lamda = lcm(p-1, q-1)
    mpz_mul(n_square, n, n);
    mpz_t lambda_power;
    mpz_init(lambda_power);
    mpz_powm(lambda_power, generator, lambda, n_square);
    L_function(mu, lambda_power, n);
    mpz_invert(mu, mu, N); // u = L((generator^lambda) mod n ^ 2) ) ^ -1 mod modulus
    mpz_clear(tmp1);
    mpz_clear(tmp2);
    mpz_clear(tmp3);
    mpz_clear(tmp4);
    mpz_clear(lambda_power);
}