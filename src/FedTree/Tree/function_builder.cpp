//
// Created by liqinbin on 12/11/20.
//

#include <FedTree/Tree/function_builder.h>
#include "FedTree/Tree/exact_tree_builder.h"
#include "FedTree/Tree/hist_tree_builder.h"

FunctionBuilder *FunctionBuilder::create(std::string name) {
    if (name == "exact") return new ExactTreeBuilder;
    if (name == "hist") return new HistTreeBuilder;
    LOG(FATAL) << "unknown builder " << name;
    return nullptr;
}

