//
// Created by liqinbin on 10/19/20.
//

#include "FedTree/FL/"
#include "gtest/gtest.h"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);
//    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Enabled, "false");
    return RUN_ALL_TESTS();
}
