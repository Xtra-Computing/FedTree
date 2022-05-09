//
// Created by liqinbin on 10/14/20.
// ThunderGBM common.h: https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/common.h
// Under Apache-2.0 license
// copyright (c) 2020 jiashuai
//

#ifndef FEDTREE_COMMON_H
#define FEDTREE_COMMON_H

#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT

#include "FedTree/util/log.h"
#include "cstdlib"
#include "config.h"
#include <math.h>
#include "thrust/tuple.h"
//#include "FedTree/Encryption/HE.h"


using std::vector;
using std::string;

#define NO_GPU \
LOG(FATAL)<<"Cannot use GPU when compiling without GPU"

//https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template<typename ... Args>
std::string string_format(const std::string &format, Args ... args) {
    size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

//data types
#ifdef USE_DOUBLE
typedef double float_type;
#else
typedef float float_type;
#endif

//CUDA macro
#ifdef USE_CUDA
    #include "cuda_runtime_api.h"
    #define CUB_IGNORE_DEPRECATED_CPP_DIALECT
    #define CUDA_CHECK(condition) \
      /* Code block avoids redefinition of cudaError_t error */ \
      do { \
        cudaError_t error = condition; \
        CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
      } while (false)


    #include "gmp.h"
    #include "FedTree/Encryption/paillier_gmp.h"


#else
#include "FedTree/Encryption/paillier.h"
#endif

#define HOST_DEVICE __host__ __device__

struct GHPair {
    float_type g;
    float_type h;
    bool encrypted = false;
#ifdef USE_CUDA
    mpz_t g_enc;
    mpz_t h_enc;
    Paillier_GMP paillier;


    void homo_encrypt(const Paillier_GMP &pl) {
        if (!encrypted) {
            mpz_t g_mpz, h_mpz;
            mpz_init(g_mpz);
            mpz_init(h_mpz);
//            if(g != 0){
            long g_ul = (long) (g * 1e6);
            mpz_import(g_mpz, 1, -1, sizeof(g_ul), 0, 0, &g_ul);
//            }
//            if(h != 0){
            long h_ul = (long) (h * 1e6);
            mpz_import(h_mpz, 1, -1, sizeof(h_ul), 0, 0, &h_ul);
//            }
            pl.encrypt(g_enc, g_mpz);
            pl.encrypt(h_enc, h_mpz);
            this->paillier = pl;
            g = 0;
            h = 0;
            encrypted = true;
            mpz_clear(g_mpz);
            mpz_clear(h_mpz);
        }
    }

    void homo_decrypt(const Paillier_GMP &pl) {
        if (encrypted) {
            mpz_t g_dec, h_dec;
            long g_l = 0, h_l = 0;
            pl.decrypt(g_dec, g_enc);
            pl.decrypt(h_dec, h_enc);
            mpz_export(&g_l, 0, -1, sizeof(g_l), 0, 0, g_dec);
            mpz_export(&h_l, 0, -1, sizeof(h_l), 0, 0, h_dec);
            g = (float_type) g_l / 1e6;
            h = (float_type) h_l / 1e6;
            encrypted = false;
        }
    }

#else
    NTL::ZZ g_enc;
    NTL::ZZ h_enc;
    Paillier paillier;

    HOST_DEVICE void homo_encrypt(const Paillier &pl) {
        if (!encrypted) {
            g_enc = pl.encrypt(NTL::to_ZZ((unsigned long) (g * 1e6)));
            h_enc = pl.encrypt(NTL::to_ZZ((unsigned long) (h * 1e6)));
            this->paillier = pl;
            g = 0;
            h = 0;
            encrypted = true;
        }
    }

    HOST_DEVICE void homo_decrypt(const Paillier &pl) {
        if (encrypted) {
//            NTL::ZZ res = pl.decrypt(g_enc);
//            std::cout<<"encrypted res:"<<res<<std::endl;
            long g_dec = NTL::to_long(pl.decrypt(g_enc));
            long h_dec = NTL::to_long(pl.decrypt(h_enc));
            g = (float_type) g_dec / 1e6;
            h = (float_type) h_dec / 1e6;
            encrypted = false;
        }
    }

#endif

    GHPair operator+(const GHPair &rhs) const {
        GHPair res;
        if (!encrypted && !rhs.encrypted) {
            res.g = this->g + rhs.g;
            res.h = this->h + rhs.h;
            res.encrypted = false;
        } else {
            if (!encrypted) {
                GHPair tmp_lhs = *this;
                tmp_lhs.homo_encrypt(rhs.paillier);
                #ifdef USE_CUDA
                rhs.paillier.add(res.g_enc, tmp_lhs.g_enc, rhs.g_enc);
                rhs.paillier.add(res.h_enc, tmp_lhs.h_enc, rhs.h_enc);
                //first transform g_enc to ntl::zz, use nlt::zz operations, then transform back to mpz_t
//                rhs.paillier.paillier_ntl.add(res.g_enc, tmp_lhs.g_enc, rhs.g_enc);
//                rhs.paillier.add(res.h_enc, tmp_lhs.h_enc, rhs.h_enc);
                #else
                res.g_enc = rhs.paillier.add(tmp_lhs.g_enc, rhs.g_enc);
                res.h_enc = rhs.paillier.add(tmp_lhs.h_enc, rhs.h_enc);
                #endif
                res.paillier = rhs.paillier;
            } else if (!rhs.encrypted) {
                GHPair tmp_rhs = rhs;
                tmp_rhs.homo_encrypt(paillier);
                #ifdef USE_CUDA
                paillier.add(res.g_enc, g_enc, tmp_rhs.g_enc);
                paillier.add(res.h_enc, h_enc, tmp_rhs.h_enc);
                #else
                res.g_enc = paillier.add(g_enc, tmp_rhs.g_enc);
                res.h_enc = paillier.add(h_enc, tmp_rhs.h_enc);
                #endif
                res.paillier = paillier;
            } else {
                #ifdef USE_CUDA
                paillier.add(res.g_enc, g_enc, rhs.g_enc);
                paillier.add(res.h_enc, h_enc, rhs.h_enc);
                #else
                res.g_enc = paillier.add(g_enc, rhs.g_enc);
                res.h_enc = paillier.add(h_enc, rhs.h_enc);
                #endif
                res.paillier = paillier;
            }
            res.encrypted = true;
        }
        return res;
    }


//#ifdef USE_CUDA
//    GHPair operator-(const GHPair &rhs) const {
//        //rewrite the function by transforming mpz_t->str->ntl::zz, and then operations, and then ntl::zz->str->mpz_t.
//        //    NTL::ZZ minus_one = NTL::to_ZZ((unsigned long) -1);
////    std::string ss;
////    ss = mpz_get_str(NULL, 10, tmp_lsh)
//    }
//#else
//
//
//#endif


    GHPair operator-(const GHPair &rhs) const {
//        std::cout<<"in operator -"<<std::endl;
        GHPair res;
        if (!encrypted && !rhs.encrypted) {
            res.g = this->g - rhs.g;
            res.h = this->h - rhs.h;
            res.encrypted = false;
        } else {
            GHPair tmp_lhs = *this;
            GHPair tmp_rhs = rhs;
#ifdef USE_CUDA
            mpz_t minus_one;
            mpz_init(minus_one);
            long mo = (long) -1;
            mpz_import(minus_one, 1, -1, sizeof(mo), 0, 0, &mo);
            if(!encrypted){
                tmp_lhs.homo_encrypt(rhs.paillier);
                mpz_t minus_g_enc, minus_h_enc;
                rhs.paillier.mul(minus_g_enc, tmp_rhs.g_enc, minus_one);
                rhs.paillier.mul(minus_h_enc, tmp_rhs.h_enc, minus_one);
                rhs.paillier.add(res.g_enc, tmp_lhs.g_enc, minus_g_enc);
                rhs.paillier.add(res.h_enc, tmp_lhs.h_enc, minus_h_enc);
                mpz_clear(minus_g_enc);
                mpz_clear(minus_h_enc);

//                mpz_invert(tmp_rhs.g_enc, tmp_rhs.g_enc, rhs.paillier.n_square);
//                mpz_invert(tmp_rhs.h_enc, tmp_rhs.h_enc, rhs.paillier.n_square);
//                rhs.paillier.add(res.g_enc, tmp_lhs.g_enc, tmp_rhs.g_enc);
//                rhs.paillier.add(res.h_enc, tmp_lhs.h_enc, tmp_rhs.h_enc);

                res.paillier = rhs.paillier;
            } else if (!rhs.encrypted) {
                tmp_rhs.g *= -1;
                tmp_rhs.h *= -1;
                tmp_rhs.homo_encrypt(paillier);
                paillier.add(res.g_enc, g_enc, tmp_rhs.g_enc);
                paillier.add(res.h_enc, h_enc, tmp_rhs.h_enc);
                res.paillier = paillier;
            } else{
                mpz_t minus_g_enc, minus_h_enc;
                mpz_init(minus_g_enc);
                mpz_init(minus_h_enc);
                paillier.mul(minus_g_enc, tmp_rhs.g_enc, minus_one);
                paillier.mul(minus_h_enc, tmp_rhs.h_enc, minus_one);
                paillier.add(res.g_enc, g_enc, minus_g_enc);
                paillier.add(res.h_enc, h_enc, minus_h_enc);
                mpz_clear(minus_g_enc);
                mpz_clear(minus_h_enc);

//                mpz_invert(tmp_rhs.g_enc, tmp_rhs.g_enc, paillier.n_square);
//                mpz_invert(tmp_rhs.h_enc, tmp_rhs.h_enc, paillier.n_square);
//                rhs.paillier.add(res.g_enc, g_enc, tmp_rhs.g_enc);
//                rhs.paillier.add(res.h_enc, h_enc, tmp_rhs.h_enc);
//
                res.paillier = paillier;
            }
            mpz_clear(minus_one);
#else
            NTL::ZZ minus_one = NTL::to_ZZ((unsigned long) -1);
            if (!encrypted) {
                tmp_lhs.homo_encrypt(rhs.paillier);
                tmp_rhs.g_enc = rhs.paillier.mul(tmp_rhs.g_enc, minus_one);
                tmp_rhs.h_enc = rhs.paillier.mul(tmp_rhs.h_enc, minus_one);
                res.g_enc = rhs.paillier.add(tmp_lhs.g_enc, tmp_rhs.g_enc);
                res.h_enc = rhs.paillier.add(tmp_lhs.h_enc, tmp_rhs.h_enc);
                res.paillier = rhs.paillier;
            } else if (!rhs.encrypted) {
                tmp_rhs.g *= -1;
                tmp_rhs.h *= -1;
                tmp_rhs.homo_encrypt(paillier);
                res.g_enc = paillier.add(g_enc, tmp_rhs.g_enc);
                res.h_enc = paillier.add(h_enc, tmp_rhs.h_enc);
                res.paillier = paillier;
            } else {
                tmp_rhs.g_enc = paillier.mul(tmp_rhs.g_enc, minus_one);
                tmp_rhs.h_enc = paillier.mul(tmp_rhs.h_enc, minus_one);
                res.g_enc = paillier.add(g_enc, tmp_rhs.g_enc);
                res.h_enc = paillier.add(h_enc, tmp_rhs.h_enc);
                res.paillier = paillier;
            }
#endif
            res.encrypted = true;
        }
        return res;
    }

    HOST_DEVICE bool operator==(const GHPair &rhs) const {
        return this->g == rhs.g && this->h == rhs.h;
    }

    HOST_DEVICE bool operator!=(const GHPair &rhs) const {
        return !(*this == rhs);
    }

    HOST_DEVICE GHPair() : g(0), h(0) {
        #ifdef USE_CUDA
        mpz_init(g_enc);
        mpz_init(h_enc);
        #endif
    };

    HOST_DEVICE GHPair(float_type v) : g(v), h(v) {
        #ifdef USE_CUDA
        mpz_init(g_enc);
        mpz_init(h_enc);
        #endif
    };

    HOST_DEVICE GHPair(float_type g, float_type h) : g(g), h(h) {
        #ifdef USE_CUDA
        mpz_init(g_enc);
        mpz_init(h_enc);
        #endif
    };

    GHPair(const GHPair& other) {
        g = other.g;
        h = other.h;
        #ifdef USE_CUDA
        mpz_init(g_enc);
        mpz_init(h_enc);
        #endif
        if(other.encrypted) {
            #ifdef USE_CUDA
            mpz_set(g_enc, other.g_enc);
            mpz_set(h_enc, other.h_enc);
            #else
            g_enc = other.g_enc;
            h_enc = other.h_enc;
            #endif
        }
        paillier = other.paillier;
        encrypted = other.encrypted;
    }

//    GHPair& operator=(GHPair& other) {
//        g = other.g;
//        h = other.h;
//        if(other.encrypted) {
//#ifdef USE_CUDA
//            mpz_init(g_enc);
//            mpz_init(h_enc);
//            mpz_set(g_enc, other.g_enc);
//            mpz_set(h_enc, other.h_enc);
//#else
//            g_enc = other.g_enc;
//            h_enc = other.h_enc;
//#endif
//        }
//        paillier = other.paillier;
//        encrypted = other.encrypted;
//        return *this;
//    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const GHPair &p) {
        os << string_format("%f/%f", p.g, p.h);
        return os;
    }
};

typedef thrust::tuple<int, float_type> int_float;

std::ostream &operator<<(std::ostream &os, const int_float &rhs);

struct BestInfo {
    int pid;
    int nid;
    int idx;
    int global_fid;
    float gain;

    friend std::ostream &operator<<(std::ostream &os, const BestInfo &best) {
        os << string_format("%d/%d/%d/%f", best.pid, best.nid, best.idx, best.gain);
        return os;
    }
};

#endif //FEDTREE_COMMON_H
