//
// Created by liqinbin on 10/14/20.
//
#include "FedTree/FL/FLtrainer.h"
#include "FedTree/Encryption/HE.h"

void FLtrainer::horizontal_fl_trainer(vector<Party> &parties, Server &server, FLParam &params){
// Is propose_split_candidates implemented? Should it be a method of TreeBuilder, HistTreeBuilder or server? Shouldnt there be a vector of SplitCandidates returned
//  vector<SplitCandidate> candidates = server.fbuilder.propose_split_candidates();
//  std::tuple <AdditivelyHE::PaillierPublicKey, AdditivelyHE::PaillierPrivateKey> key_pair = server.HE.generate_key_pairs();
//    server.send_info(parties, std::get<0>(keyPairs), candidates);
//    for (int i = 0; i < params.gbdt_param.n_trees; i++){
//        for (j = 0; j < parties.size(); j++){
//            parties[j].update_gradients();
//        }
//        for (int j = 0; j < params.gbdt_param.depth; j++){
//            for (int k = 0; k < parties.size(); k++) {
//                SyncArray<GHPair> hist = parties[j].fbuilder->compute_histogram();
//                if (params.privacy_tech == "he") {
                    // Should HE be a public member of Party?
//                    parties[k].HE.encryption();
//                }
//                if (params.privacy_tech == "dp") {
                    // Should DP be public member of Party?
//                    parties[k].DP.add_gaussian_noise();
//                }
//                parties[k].send_info(hist);
//            }
            // merge_histograms in tree_builder?
//            server.sum_histograms(); // or on Party 1 if using homo encryption
//            server.HE.decrption();
//            if (j != params.gbdt_param.depth - 1) {
//                server.fbuilder.compute_gain();
//                server.fbuilder.get_best_split(); // or using exponential mechanism
//                server.fbuilder.update_tree();
//                server.send_info(); // send split points
//            }
//            else{
//                server.fbuilder.compute_leaf_value();
//                server.DP.add_gaussian_noise(); // for DP: add noises to the tree
//                server.send_info(); // send leaf values
//            }
//        }
//    }
//}


void FLtrainer::vertical_fl_trainer(vector<Party> &parties, Server &server, FLParam &params){
    parties[0].homo_init();
//    parties[0].send_info();
//    for (int i = 0; i < params.gbdt_param.n_trees; i++){
//        parties[0].update_gradients();
//        parties[0].DP.add_gaussian_noise();
//        parties[0].send_info();
//        for (int j = 0; j < params.gbdt_param.depth; j++){
//            if (j != params.gbdt_param.depth - 1){
//                for (int k = 0; k < parties.size(); k++){
//                    parties[k].fbuilder.propose_split_candidates();
//                    parties[k].fbuilder.compute_histogram();
//                    parties[k].HE.encrypt();
//                }
//                parties[0].decrypt();
//                parties[0].compute_gain();
//                parties[0].get_best_split();
//                parties[0].send_info();
//                parties[z].fbuilder.update_tree();
//                parties[z].send_info()
//            }
//            else{
//                parties[0].fbuilder.compute_leaf_value();
//                parties[0].DP.add_gaussian_noise();
//            }
//        }
//    }
}