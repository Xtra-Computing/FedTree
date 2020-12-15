//
// Created by liqinbin on 12/11/20.
//

#include "FedTree/trainer.h"
#include "FedTree/metric/metric.h"
#include "FedTree/util/device_lambda.cuh"
#include "thrust/reduce.h"
#include "FedTree/booster.h"
#include "FedTree/parser.h"

#include "fstream"
#include "chrono"
#include "time.h"
#include "thrust/reduce.h"

using namespace std;

vector<vector<Tree>> 