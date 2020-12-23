//
// Created by hanyuxuan on 22/10/20.
//
#include "FedTree/common.h"
#include <random>

#ifndef FEDTREE_DIRICHLET_H
#define FEDTREE_DIRICHLET_H

template<class RNG>
class dirichlet_distribution {
public:
    dirichlet_distribution(const vector<float> &);

    void set_params(const vector<float> &);

    vector<float> get_params();

    vector<float> operator()(RNG &);

private:
    vector<float> alpha;
    vector<std::gamma_distribution<>> gamma;
};

template<class RNG>
dirichlet_distribution<RNG>::dirichlet_distribution(const vector<float> &alpha) {
    set_params(alpha);
}

template<class RNG>
void dirichlet_distribution<RNG>::set_params(const vector<float> &new_params) {
    alpha = new_params;
    vector<std::gamma_distribution<>> new_gamma(alpha.size());
    for (int i = 0; i < alpha.size(); ++i) {
        std::gamma_distribution<> temp(alpha[i], 1);
        new_gamma[i] = temp;
    }
    gamma = new_gamma;
}

template<class RNG>
vector<float> dirichlet_distribution<RNG>::get_params() {
    return alpha;
}

template<class RNG>
vector<float> dirichlet_distribution<RNG>::operator()(RNG &generator) {
    vector<float> x(alpha.size());
    float sum = 0.0;
    for (int i = 0; i < alpha.size(); ++i) {
        x[i] = gamma[i](generator);
        sum += x[i];
    }
    for (float &xi : x) xi = xi / sum;
    return x;
}

#endif //FEDTREE_DIRICHLET_H
