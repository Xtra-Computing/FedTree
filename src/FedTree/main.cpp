//
// Created by liqinbin on 10/13/20.
//

#include "FedTree/FL/FLParams.h"
#include "FedTree/FL/FLtrainer.h"
#include "FedTree/parser.h"
#include "FedTree/dataset.h"

int main(int argc, char** argv){
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);

    //initialize parameters
    FLParams fl_param;
    Parser parser;
    parser.parse_param(fl_param, argc, argv);

    //load dataset from file/files
    Dataset dataset;
    dataset.load_from_file(fl_param.dataset_path);

    //initialize parties and server *with the dataset*
    vector<Party> parties;
    for(i = 0; i < fl_param.n_parties; i++){
        Party party;
        parties.push_back(party);
    }
    Server server;

    //train
    FLtrainer trainer;
    model = trainer.train(parties, server, fl_param);

    //test
    Dataset test_dataset;
    test_dataset.load_from_file(fl_param.test_dataset_path);
    acc = model.predict(test_dataset);
    return 0;
}