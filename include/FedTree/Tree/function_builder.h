//
// Created by liqinbin on 12/11/20.
//

#ifndef FEDTREE_FUNCTION_BUILDER_H
#define FEDTREE_FUNCTION_BUILDER_H


#include <FedTree/tree.h>
#include "FedTree/common.h"
#include "FedTree/sparse_columns.h"

class FunctionBuilder {
public:
    virtual vector<Tree> build_approximate(const SyncArray<GHPair> &gradients) = 0;

    virtual void init(const DataSet &dataset, const GBMParam &param) {
        this->param = param;
    };

    virtual const SyncArray<float_type> &get_y_predict(){ return y_predict; };

    virtual ~FunctionBuilder(){};

    static FunctionBuilder *create(std::string name);

protected:
    SyncArray<float_type> y_predict;
    GBMParam param;
};


#endif //FEDTREE_FUNCTION_BUILDER_H
