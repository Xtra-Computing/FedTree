//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_SERVER_H
#define FEDTREE_SERVER_H

#include "FedTree/dataset.h"
#include "FedTree/Tree/tree_builder.h"
#include "FedTree/Encryption/HE.h"
#include "FedTree/DP/noises.h"

// Todo: the server structure.

class Server {
    void propose_split_candidates();
    void send_info(string info_type);
    void sum_histograms();

private:
    DataSet dataset;
    TreeBuilder fbuilder;
    AdditivelyHE HE;
    DPnoises<double> DP;
};

#endif //FEDTREE_SERVER_H
