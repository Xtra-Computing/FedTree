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
#include "thrust/tuple.h"
//#include "FedTree/Encryption/HE.h"
#include "FedTree/Encryption/paillier.h"

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
typedef float float_type;

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

#endif

#define HOST_DEVICE __host__ __device__

struct GHPair {
    float_type g;
    float_type h;
    NTL::ZZ g_enc;
    NTL::ZZ h_enc;
    Paillier paillier;
    bool encrypted = false;

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
            long g_dec = NTL::to_long(pl.decrypt(g_enc));
            long h_dec = NTL::to_long(pl.decrypt(h_enc));
            g = (float_type) g_dec / 1e6;
            h = (float_type) h_dec / 1e6;
            encrypted = false;
        }
    }

    HOST_DEVICE GHPair operator+(const GHPair &rhs) const {
        GHPair res;
        if (!encrypted && !rhs.encrypted) {
            res.g = this->g + rhs.g;
            res.h = this->h + rhs.h;
            res.encrypted = false;
        } else {
            if (!encrypted) {
                GHPair tmp_lhs = *this;
                tmp_lhs.homo_encrypt(rhs.paillier);
                res.g_enc = rhs.paillier.add(tmp_lhs.g_enc, rhs.g_enc);
                res.h_enc = rhs.paillier.add(tmp_lhs.h_enc, rhs.h_enc);
                res.paillier = rhs.paillier;
            } else if (!rhs.encrypted) {
                GHPair tmp_rhs = rhs;
                tmp_rhs.homo_encrypt(paillier);
                res.g_enc = paillier.add(g_enc, tmp_rhs.g_enc);
                res.h_enc = paillier.add(h_enc, tmp_rhs.h_enc);
                res.paillier = paillier;
            } else {
                res.g_enc = paillier.add(g_enc, rhs.g_enc);
                res.h_enc = paillier.add(h_enc, rhs.h_enc);
                res.paillier = paillier;
            }
            res.encrypted = true;
        }
        return res;
    }

    HOST_DEVICE GHPair operator-(const GHPair &rhs) const {
        GHPair res;
        if (!encrypted && !rhs.encrypted) {
            res.g = this->g - rhs.g;
            res.h = this->h - rhs.h;
            res.encrypted = false;
        } else {
            GHPair tmp_lhs = *this;
            GHPair tmp_rhs = rhs;
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

    HOST_DEVICE GHPair() : g(0), h(0) {};

    HOST_DEVICE GHPair(float_type v) : g(v), h(v) {};

    HOST_DEVICE GHPair(float_type g, float_type h) : g(g), h(h) {};

    GHPair(const GHPair& other) {
        g = other.g;
        h = other.h;
        g_enc = other.g_enc;
        h_enc= other.h_enc;
        paillier = other.paillier;
        encrypted = other.encrypted;
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const GHPair &p) {
        os << string_format("%f/%f", p.g, p.h);
        return os;
    }
};

typedef thrust::tuple<int, float_type> int_float;

std::ostream &operator<<(std::ostream &os, const int_float &rhs);


#endif //FEDTREE_COMMON_H
