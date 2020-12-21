//
// Created by liqinbin on 12/15/20.
//

#ifndef FEDTREE_RANKING_OBJ_H
#define FEDTREE_RANKING_OBJ_H

#include "objective_function.h"

/**
 *
 * https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
 */
class LambdaRank : public ObjectiveFunction {
public:
    void get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p,
                      SyncArray<GHPair> &gh_pair) override;

    void configure(GBDTParam param, const DataSet &dataset) override;

    string default_metric_name() override;

    virtual ~LambdaRank() override = default;

protected:
    virtual inline float_type get_delta_z(float_type labelI, float_type labelJ, int rankI, int rankJ, int group_id) { return 1; };

    vector<int> gptr;//group start position
    int n_group;

    float_type sigma;
};

class LambdaRankNDCG : public LambdaRank {
public:
    void configure(GBDTParam param, const DataSet &dataset) override;

    string default_metric_name() override;

protected:
    float_type get_delta_z(float_type labelI, float_type labelJ, int rankI, int rankJ, int group_id) override;

private:
    vector<float_type> idcg;
};



#endif //FEDTREE_RANKING_OBJ_H
