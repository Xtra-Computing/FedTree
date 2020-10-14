//
// Created by liqinbin on 10/14/20.
// ThunderGBM common.h: https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/common.h
// Under Apache-2.0 license
// copyright (c) 2020 jiashuai
//

#ifndef FEDTREE_COMMON_H
#define FEDTREE_COMMON_H

#include "FedTree/util/log.h"
#include "cuda_runtime_api.h"
#include "cstdlib"
#include "config.h"
#include "thrust/tuple.h"

using std::vector;
using std::string;

//CUDA macro
#define USE_CUDA
#define NO_GPU \
LOG(FATAL)<<"Cannot use GPU when compiling without GPU"
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (false)

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

#define HOST_DEVICE __host__ __device__

struct GHPair {
    float_type g;
    float_type h;

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
