//
// Created by liqinbin on 12/11/20.
//

#ifndef FEDTREE_FUNCTION_BUILDER_H
#define FEDTREE_FUNCTION_BUILDER_H


#include "tree.h"
#include "FedTree/common.h"
#include "FedTree/dataset.h"
#include "FedTree/Encryption/HE.h"

class FunctionBuilder {
public:
    virtual vector<Tree> build_approximate(const SyncArray<GHPair> &gradients) = 0;

    virtual void build_init(const SyncArray<GHPair> &gradients, int k) = 0;

    virtual void compute_hist(int level) = 0;

    virtual void init(DataSet &dataset, const GBDTParam &param) {
        this->param = param;
    };

    virtual const SyncArray<float_type> &get_y_predict() { return y_predict; };

    virtual ~FunctionBuilder() {};

    static FunctionBuilder *create(std::string name);

protected:
    SyncArray<float_type> y_predict;
    GBDTParam param;
};


#endif //FEDTREE_FUNCTION_BUILDER_H
