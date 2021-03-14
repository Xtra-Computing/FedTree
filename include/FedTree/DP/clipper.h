//
// Created by Tianyuan Fu on 10/3/21.
//

#ifndef FEDTREE_CLIPPER_H
#define FEDTREE_CLIPPER_H

#include <algorithm>
using namespace std;
template <typename T>
class DPClipper {
public:
    static void clip_gradient_value(T& value) {
        value = max<T>(min<T>(value, 1),-1);
    }
};
#endif //FEDTREE_CLIPPER_H
