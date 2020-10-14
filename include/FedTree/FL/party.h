//
// Created by liqinbin on 10/13/20.
//

#ifndef FEDTREE_PARTY_H
#define FEDTREE_PARTY_H

// Todo: the party structure
class Party {
public:
    void init(int pid, const DataSet &dataSet) {
        this->pid = pid;
        this->dataset = dataset;
    };
    int pid;
private:
    DataSet dataSet;
};

#endif //FEDTREE_PARTY_H
