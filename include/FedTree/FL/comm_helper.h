//
// Created by liqinbin on 12/23/20.
//

#ifndef FEDTREE_COMM_HELPER_H
#define FEDTREE_COMM_HELPER_H

#include "party.h"
#include "server.h"

class Comm {
public:
    void send_last_trees_to_server(Party &party, int pid, Server &server) {
        if (server.local_trees[pid].trees.size())
            server.local_trees[pid].trees[0] = party.gbdt.trees.back();
        else
            server.local_trees[pid].trees.push_back(party.gbdt.trees.back());
    }

    void send_last_global_trees_to_party(Server &server, Party &party) {
        party.gbdt.trees.push_back(server.global_trees.trees.back());
    };

    void send_all_trees_to_server(Party &party, int pid, Server &server) {
        server.local_trees[pid].trees = party.gbdt.trees;
    }

    template<class T>
    SyncArray<T> concat_msyncarray(MSyncArray<T> &arrays, vector<int> parties_n_bins, int n_nodes_in_level) {
        int n_bins_sum = accumulate(parties_n_bins.begin(), parties_n_bins.end(), 0);
        int n_parties = parties_n_bins.size();
        SyncArray<T> concat_array(n_bins_sum * n_nodes_in_level);
        auto concat_array_data = concat_array.host_data();
        for (int i = 0; i < n_nodes_in_level; i++) {
            for (int j = 0; j < n_parties; j++) {
                auto array_data = arrays[j].host_data();
                for (int k = 0; k < parties_n_bins[j]; k++) {
                    concat_array_data[i * n_bins_sum + accumulate(parties_n_bins.begin(), parties_n_bins.begin() + j, 0) + k]
                        = array_data[i * parties_n_bins[j] + k];
                }
            }
        }
        return concat_array;
    }
};


#endif //FEDTREE_COMM_HELPER_H
