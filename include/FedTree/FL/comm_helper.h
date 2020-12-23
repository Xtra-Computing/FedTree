//
// Created by liqinbin on 12/23/20.
//

#ifndef FEDTREE_COMM_HELPER_H
#define FEDTREE_COMM_HELPER_H

#include "party.h"
#include "server.h"

class Comm {
public:
    void send_last_trees_to_server(Party &party, int pid, Server &server){
        server.local_trees[pid].trees[0] = party.gbdt.trees.back();
    }
    void send_last_global_trees_to_party(Server &server, Party &party) {
        party.gbdt.trees.push_back(server.global_trees.trees.back());
    };
};


#endif //FEDTREE_COMM_HELPER_H
