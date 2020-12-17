//
// Created by liqinbin on 12/11/20.
//

#include "FedTree/Tree/function_builder.h"
#include "FedTree/Tree/hist_tree_builder.h"

FunctionBuilder *FunctionBuilder::create(std::string name) {
    if (name == "exact") {
        std::cout<<"not supported yet";
        exit(1);
    }
    if (name == "hist") return new HistTreeBuilder;
    LOG(FATAL) << "unknown builder " << name;
    return nullptr;
}

