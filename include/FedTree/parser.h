//
// Created by liqinbin on 10/13/20.
// Edited by Tianyuan Fu on 10/19/20
//

#ifndef FEDTREE_PARSER_H
#define FEDTREE_PARSER_H
#include "FedTree/common.h"

#include <FedTree/FL/FLparam.h>
#include "dataset.h"
#include "Tree/tree.h"

// Todo: parse the parameters to FLparam. refer to ThunderGBM parser.h https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/parser.h
class Parser {
public:
    void parse_param(FLParam &fl_param, char *argv);
    void load_model(string model_path, GBDTParam &model_param, vector<vector<Tree>> &boosted_model);
    void save_model(string model_path, GBDTParam &model_param, vector<vector<Tree>> &boosted_model);
};

#endif //FEDTREE_PARSER_H
