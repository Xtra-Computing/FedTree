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

    virtual void init(DataSet &dataset, const GBDTParam &param) {
        this->param = param;
    };

    virtual const SyncArray<float_type> &get_y_predict(){ return y_predict; };

    virtual void encrypt_gradients(AdditivelyHE::PaillierPublicKey pk) = 0;

    virtual void set_gradients(SyncArray<GHPair> &gh) = 0;

    virtual SyncArray<GHPair> get_gradients() = 0;

    virtual ~FunctionBuilder(){};

    static FunctionBuilder *create(std::string name);

protected:
    SyncArray<float_type> y_predict;
    GBDTParam param;
};


#endif //FEDTREE_FUNCTION_BUILDER_H
