#include "FedTree/Tree/tree_builder.h"

float_type TreeBuilder:: compute_gain(GHPair father, GHPair lch, GHPair rch, float_type lambda) {
    return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda) -
           (father.g * father.g) / (father.h + lambda);
}