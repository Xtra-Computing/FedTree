//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_PARTY_H
#define FEDTREE_PARTY_H

#include "FedTree/dataset.h"
#include "FedTree/Tree/tree_builder.h"
#include "FedTree/Encryption/HE.h"
#include "FedTree/DP/noises.h"

// Todo: the party structure
class Party {
public:
    void init(int pid, const DataSet &dataSet) {
        this->pid = pid;
        this->dataset = dataset;
    };
    void homo_encrytion();
    int pid;
private:
    DataSet dataset;
    std::unique_ptr<TreeBuilder> fbuilder;
    AdditivelyHE HE;
    DPnoises<double> DP;

};

#endif //FEDTREE_PARTY_H
