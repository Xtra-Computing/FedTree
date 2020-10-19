//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_NOISES_H
#define FEDTREE_NOISES_H

#include <random>

template <typename T>
void add_gaussian_noise(T& data, float variance) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, variance);

    double noise = distribution(generator);
    *data += noise;
}

template <typename T>
void add_laplasian_noise(T& data, float variance) {
    // a r.v. following Laplace(0, b) is equivalent to the difference of 2 i.i.d Exp(1/b) r.v.
    double b = sqrt(variance/2);
    std::default_random_engine generator;
    std::exponential_distribution<double> distribution(1/b);
    double noise = distribution(generator) - distribution(generator);
    *data += noise;
}
#endif //FEDTREE_NOISES_H
