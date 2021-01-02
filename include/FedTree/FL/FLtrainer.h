//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_FLTRAINER_H
#define FEDTREE_FLTRAINER_H
#include "FedTree/common.h"
#include "FedTree/FL/party.h"
#include "FedTree/FL/server.h"
// Todo: different federated training algorithms including horizontal GBDT and vertical GBDT.

class FLtrainer {
public:
    void horizontal_fl_trainer(vector<Party> &parties, Server &server, FLParam &params);

    void vertical_fl_trainer(vector<Party> &parties, Server &server, FLParam &params);

    void hybrid_fl_trainer(vector<Party> &parties, Server &server, FLParam &params);

    void ensemble_trainer(vector<Party> &parties, Server &server, FLParam &params);

};
#endif //FEDTREE_FLTRAINER_H
