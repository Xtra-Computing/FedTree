//
// Created by liqinbin on 10/14/20.
//

#include "FedTree/Tree/gbdt.h"
#include "FedTree/booster.h"

void GBDT::train(GBDTParam &param, const DataSet &dataset){
    if (param.tree_method == "auto")
        param.tree_method = "hist";
    else if (param.tree_method != "hist"){
        std::cout<<"FedTree only supports histogram-based training yet";
        exit(1);
    }

    if(param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos) {
        int num_class = dataset.label.size();
        if (param.num_class != num_class) {
            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
            param.num_class = num_class;
        }
        if(param.num_class > 2)
            param.tree_per_rounds = param.num_class;
    }
    else if(param.objective.find("reg:") != std::string::npos){
        param.num_class = 1;
    }

    Booster booster;
    booster.init(dataset, param);
}