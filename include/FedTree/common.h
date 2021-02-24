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
#include "FedTree/Encryption/HE.h"

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
    AdditivelyHE::EncryptedNumber g_enc;
    AdditivelyHE::EncryptedNumber h_enc;
    AdditivelyHE::PaillierPublicKey pk;
    bool encrypted = false;

    HOST_DEVICE void decrypt(AdditivelyHE::PaillierPrivateKey privateKey) {
        g = AdditivelyHE::decrypt(privateKey, g_enc);
        h = AdditivelyHE::decrypt(privateKey, h_enc);
    }

    HOST_DEVICE void homo_encrypt(AdditivelyHE::PaillierPublicKey pk) {
        g_enc = AdditivelyHE::encrypt(pk, g);
        h_enc = AdditivelyHE::encrypt(pk, h);
        encrypted = true;
        this->pk = pk;
        g = 0;
        h = 0;
    }

    HOST_DEVICE GHPair homo_add(const GHPair &rhs) const {
//        CHECK_EQ(this->pk, rhs.pk);
        GHPair res;
        res.homo_encrypt(pk);
        res.g_enc = AdditivelyHE::aggregate(this->g_enc, res.g_enc);
        res.h_enc = AdditivelyHE::aggregate(this->h_enc, res.h_enc);
        return res;
    }

    HOST_DEVICE GHPair operator+(const GHPair &rhs) const {
        GHPair res;
        res.g = this->g + rhs.g;
        res.h = this->h + rhs.h;
        return res;
    }

    HOST_DEVICE const GHPair operator-(const GHPair &rhs) const {
        GHPair res;
        res.g = this->g - rhs.g;
        res.h = this->h - rhs.h;
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

    friend std::ostream &operator<<(std::ostream &os,
                                    const GHPair &p) {
        os << string_format("%f/%f", p.g, p.h);
        return os;
    }
};

typedef thrust::tuple<int, float_type> int_float;

std::ostream &operator<<(std::ostream &os, const int_float &rhs);


#endif //FEDTREE_COMMON_H
