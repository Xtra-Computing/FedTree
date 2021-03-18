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

    HOST_DEVICE void homo_encrypt(AdditivelyHE::PaillierPublicKey pk) {
        g_enc = AdditivelyHE::encrypt(pk, g);
        h_enc = AdditivelyHE::encrypt(pk, h);
        this->pk = pk;
        g = 0;
        h = 0;
        encrypted = true;
    }

    HOST_DEVICE void homo_decrypt(AdditivelyHE::PaillierPrivateKey privateKey) {
        if (encrypted) {
            g = AdditivelyHE::decrypt(privateKey, g_enc);
            h = AdditivelyHE::decrypt(privateKey, h_enc);
            encrypted = false;
        }
//        else {
//            LOG(INFO) << "Trying to decrypt unencrypted numbers.";
//        }
    }

    HOST_DEVICE GHPair operator+(const GHPair &rhs) const {
        GHPair res;
        if (encrypted && rhs.encrypted) {
            res.g_enc = AdditivelyHE::aggregate(this->g_enc, rhs.g_enc);
            res.h_enc = AdditivelyHE::aggregate(this->h_enc, rhs.h_enc);
            res.pk = this->pk;
            res.encrypted = true;
        } else if (!encrypted && rhs.encrypted) {
            res.g_enc = AdditivelyHE::aggregate_scalar(rhs.g_enc, this->g);
            res.h_enc = AdditivelyHE::aggregate_scalar(rhs.h_enc, this->h);
            res.pk = rhs.pk;
            res.encrypted = true;
        } else if (encrypted && !rhs.encrypted) {
            res.g_enc = AdditivelyHE::aggregate_scalar(this->g_enc, rhs.g);
            res.h_enc = AdditivelyHE::aggregate_scalar(this->h_enc, rhs.h);
            res.pk = this->pk;
            res.encrypted = true;
        } else {
            res.g = this->g + rhs.g;
            res.h = this->h + rhs.h;
            res.encrypted = false;
        }
        return res;
    }

    HOST_DEVICE GHPair operator-(const GHPair &rhs) const {
        GHPair res;
        if (encrypted && rhs.encrypted) {
            res.g_enc = AdditivelyHE::aggregate(this->g_enc, AdditivelyHE::multiply_scalar(rhs.g_enc, -1));
            res.h_enc = AdditivelyHE::aggregate(this->h_enc, AdditivelyHE::multiply_scalar(rhs.h_enc, -1));
            res.pk = this->pk;
            res.encrypted = true;
        } else if (!encrypted && rhs.encrypted) {
            res.g_enc = AdditivelyHE::aggregate_scalar(AdditivelyHE::multiply_scalar(rhs.g_enc, -1), this->g);
            res.h_enc = AdditivelyHE::aggregate_scalar(AdditivelyHE::multiply_scalar(rhs.h_enc, -1), this->h);
            res.pk = rhs.pk;
            res.encrypted = true;
        } else if (encrypted && !rhs.encrypted) {
            res.g_enc = AdditivelyHE::aggregate_scalar(this->g_enc, -rhs.g);
            res.h_enc = AdditivelyHE::aggregate_scalar(this->h_enc, -rhs.h);
            res.pk = this->pk;
            res.encrypted = true;
        } else {
            res.g = this->g - rhs.g;
            res.h = this->h - rhs.h;
            res.encrypted = false;
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

    friend std::ostream &operator<<(std::ostream &os,
                                    const GHPair &p) {
        os << string_format("%f/%f", p.g, p.h);
        return os;
    }
};

typedef thrust::tuple<int, float_type> int_float;

std::ostream &operator<<(std::ostream &os, const int_float &rhs);


#endif //FEDTREE_COMMON_H
