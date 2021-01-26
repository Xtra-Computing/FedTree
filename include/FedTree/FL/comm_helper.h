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
        if(server.local_trees[pid].trees.size())
            server.local_trees[pid].trees[0] = party.gbdt.trees.back();
        else
            server.local_trees[pid].trees.push_back(party.gbdt.trees.back());
    }
    void send_last_global_trees_to_party(Server &server, Party &party) {
        party.gbdt.trees.push_back(server.global_trees.trees.back());
    };
    void send_all_trees_to_server(Party &party, int pid, Server &server){
        server.local_trees[pid].trees = party.gbdt.trees;
    }

    template<class T>
    SyncArray<T> concat_msyncarray(MSyncArray<T> &arrays) {
        int total_size = 0;
        vector<int> ptr = {0};
        for (int i = 0; i < arrays.size(); i++) {
            total_size += arrays[i].size();
            ptr.push_back(ptr.back() + total_size);
        }
        SyncArray<T> concat_array(total_size);
        auto concat_array_data = concat_array.host_data();

        for (int i = 0; i < arrays.size(); i++) {
            auto array_data = arrays[i].host_data();
            for (int j = 0; j < arrays[i].size(); j++) {
                concat_array_data[ptr[i] + j] = array_data[j];
            }
        }
        return concat_array;
    }
};


#endif //FEDTREE_COMM_HELPER_H
