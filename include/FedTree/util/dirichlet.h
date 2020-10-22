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
    dirichlet_distribution(const vector<double> &);

    void set_params(const vector<double> &);

    vector<double> get_params();

    vector<double> operator()(RNG &);

private:
    vector<double> alpha;
    vector<std::gamma_distribution<>> gamma;
};

template<class RNG>
dirichlet_distribution<RNG>::dirichlet_distribution(const vector<double> &alpha) {
    set_params(alpha);
}

template<class RNG>
void dirichlet_distribution<RNG>::set_params(const vector<double> &new_params) {
    alpha = new_params;
    vector<std::gamma_distribution<>> new_gamma(alpha.size());
    for (int i = 0; i < alpha.size(); ++i) {
        std::gamma_distribution<> temp(alpha[i], 1);
        new_gamma[i] = temp;
    }
    gamma = new_gamma;
}

template<class RNG>
vector<double> dirichlet_distribution<RNG>::get_params() {
    return alpha;
}

template<class RNG>
vector<double> dirichlet_distribution<RNG>::operator()(RNG &generator) {
    vector<double> x(alpha.size());
    double sum = 0.0;
    for (int i = 0; i < alpha.size(); ++i) {
        x[i] = gamma[i](generator);
        sum += x[i];
    }
    for (double &xi : x) xi = xi / sum;
    return x;
}

#endif //FEDTREE_DIRICHLET_H
