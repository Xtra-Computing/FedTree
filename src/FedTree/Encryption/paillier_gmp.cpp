#include "FedTree/Encryption/paillier_gmp.h"
#include <iostream>
#include <sstream>

Paillier_GMP::Paillier_GMP(){
    mpz_init(n);
    mpz_init(n_square);
    mpz_init(generator);
    mpz_init(p);
    mpz_init(q);
    mpz_init(lambda);
    mpz_init(mu);

//    mpz_init(r);
}
void Paillier_GMP::add(mpz_t &result, const mpz_t &x, const mpz_t &y) const{
    mpz_init(result);
    mpz_mul(result, x, y);
    mpz_mod(result, result, n_square);
    return;
}


void Paillier_GMP::mul(mpz_t &result, const mpz_t &x, const mpz_t &y) const{
    mpz_init(result);
    mpz_powm(result, x, y, n_square);
    return ;
}

void Paillier_GMP::L_function(mpz_t &result, mpz_t &input, const mpz_t &N) const{
//    mpz_init(result);
    mpz_sub_ui(result, input, 1);
    mpz_tdiv_q(result, result, N);
    return;
}

void Paillier_GMP::encrypt(mpz_t &result, const mpz_t &message) const{


    gmp_randstate_t state;
    gmp_randinit_mt(state);
//    gmp_randseed_ui(state, 1000U);
    mpz_t r;
    mpz_init(r);
//    mpz_urandomb(r, state, key_length/2);
//    mpz_add_ui(r, r, 1);
//    mpz_mod(r, r, n);
    while(true) {
        mpz_urandomm(r, state, n);
        if(mpz_cmp_ui(r, 0))
            break;
    }


    mpz_t tmp;
    mpz_init(tmp);
    mpz_init(result);
    mpz_powm(result, r, n, n_square);
    mpz_powm(tmp, generator, message, n_square);
    mpz_mul(result, result, tmp);
    mpz_mod(result, result, n_square);
    mpz_clear(tmp);
    mpz_clear(r);


    //since g=1+n, g^m=(1+n)^m=1+nm % n^2
//    mpz_mul(r, message, n);
//    mpz_add_ui(r, r, 1);
//    mpz_mul(result, result, r);
//    mpz_mod(result, result, n_square);

    return;
}

void Paillier_GMP::decrypt(mpz_t &result, const mpz_t &message) const{
    mpz_init(result);
    mpz_powm(result, message, lambda, n_square);
    L_function(result, result, n);
    mpz_mul(result, result, mu);
    mpz_mod(result, result, n);
//    result_l = mpz_get_si(result);
//    mpz_export(&result_l, (size_t*)0, -1, sizeof(result_l), 0, 0, result);
//    mpz_clear(result);
    return;
}

//int gen_prime(mpz_t prime, mp_bitcnt_t len) {
//    mpz_t rnd;
//
//    mpz_init(rnd);
//
//    gen_random(rnd, len);
//
//    //set most significant bit to 1
//    mpz_setbit(rnd, len-1);
//    //look for next prime
//    mpz_nextprime(prime, rnd);
//
//    mpz_clear(rnd);
//    return 0;
//}

//int getMu(mpz_t mu, const mpz_t lambda, const mpz_t g, const mpz_t N,
//          const mpz_t N2) {
//    // µ = ( L(g^λ mod N^2) )^-1 mod N
//}

void Paillier_GMP::keyGen(uint32_t keyLength) {
    /*
    this->key_length = keyLength;

    paillier_ntl.keygen(512);
    std::stringstream ss;
    ss<<paillier_ntl.modulus;
    mpz_set_str(n, ss.str().c_str(), 10);
    ss.str("");
    mpz_mul(n_square, n, n);
    ss<<paillier_ntl.generator;
    mpz_set_str(generator, ss.str().c_str(), 10);
    ss.str("");
    ss<<paillier_ntl.lambda;
    mpz_set_str(lambda, ss.str().c_str(), 10);
    ss.str("");
    ss<<paillier_ntl.u;
    mpz_set_str(mu, ss.str().c_str(), 10);
    ss.str("");
    ss<<paillier_ntl.random;
    mpz_set_str(r, ss.str().c_str(), 10);
        */


    this->key_length = keyLength;
    gmp_randstate_t state;
    gmp_randinit_mt(state);
//    gmp_randseed_ui(state, 1000U);


//    while(true){
//        mpz_urandomb(p, state, key_length / 4);
//        mpz_urandomb(q, state, key_length / 4);
//        mpz_nextprime(p, p);
//        mpz_nextprime(q, q);
//        if (mpz_sizeinbase(p, 2) == mpz_sizeinbase(q, 2))
//            // Same bit-length
//            break;
//    }
//    mpz_mul(n,p,q);

//    do{
//        do
//            mpz_urandomb(p, state, key_length / 4);
//        while( !mpz_probab_prime_p(p, 10) );
//
//        do
//            mpz_urandomb(q, state, key_length / 4);
//        while( !mpz_probab_prime_p(q, 10) );
//
//
//
//        mpz_mul(n, p, q);
//    } while( !mpz_tstbit(n, key_length - 1) );


    mpz_t tmp1, tmp2, tmp3, tmp4;
    mpz_init(tmp1);
    mpz_init(tmp2);
    mpz_init(tmp3);
    mpz_init(tmp4);

    while (true){
        mpz_urandomb(p, state, key_length/4);
        mpz_urandomb(q, state, key_length/4);
        mpz_nextprime(p, p);
        mpz_nextprime(q, q);
        if (mpz_sizeinbase(p, 2) == mpz_sizeinbase(q, 2)) {
            mpz_sub_ui(tmp1, p, 1);
            mpz_sub_ui(tmp2, q, 1);
            mpz_mul(tmp3, tmp1, tmp2); // tmp3 = (p-1)(q-1)
            mpz_mul(tmp4, p, q); // tmp4 = p*q
            mpz_gcd(tmp3, tmp3, tmp4); // tmp = gcd(pq, (p-1)(q-1))
            if (mpz_cmp_ui(tmp3, 1) == 0) // gcd(pq, (p-1)(q-1)) == 1
                break;
        }
    }
    mpz_set(n, tmp4);      // n = p * q

    mpz_add_ui(generator, n, 1);  // g = n + 1
    mpz_sub_ui(p, p, 1);
    mpz_sub_ui(q, q, 1);
    mpz_lcm(lambda, p, q);   // lamda = lcm(p-1, q-1)

    mpz_mul(n_square, n, n);

//    while(true){
//        std::cout<<"1"<<std::endl;
//        mpz_urandomb(p, state, key_length);
//
//        mpz_powm(mu, generator, lambda, n_square);
//        L_function(mu, mu, n);
//        if(mpz_invert(mu, mu, n))
//            break;
//    }
    mpz_t lambda_power;
    mpz_init(lambda_power);
    mpz_powm(lambda_power, generator, lambda, n_square);
    L_function(mu, lambda_power, n);
    if(mpz_invert(mu, mu, n) == 0) {
        std::cout<<"wrong mu"<<std::endl;
        exit(1);
    } // u = L((generator^lambda) mod n ^ 2) ) ^ -1 mod modulus

    mpz_clear(tmp1);
    mpz_clear(tmp2);
    mpz_clear(tmp3);
    mpz_clear(tmp4);
    mpz_clear(lambda_power);





    /*
    gmp_randstate_t state;
    gmp_randinit_mt(state);

//    gmp_randseed_ui(state, 1000U);
//    mpz_t r;
    mpz_init(r);
//    mpz_urandomb(r, state, key_length/2);
//    mpz_add_ui(r, r, 1);
//    mpz_mod(r, r, n);
    while(true) {
        mpz_urandomm(r, state, n);
        if(mpz_cmp_ui(r, 0))
            break;
    }
    std::cout<<"random:"<<r<<std::endl;
     */
}