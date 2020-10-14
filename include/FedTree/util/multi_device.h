//
// Created by liqinbin on 10/14/20.
// ThunderGBM multi_device.h: https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/util/multi_device.h
// Under Apache-2.0 license
// copyright (c) 2020 jiashuai
//

#ifndef FEDTREE_MULTI_DEVICE_H
#define FEDTREE_MULTI_DEVICE_H

#include "FedTree/common.h"

//switch to specific device and do something, then switch back to the original device
//FIXME make this macro into a function?
#define DO_ON_DEVICE(device_id, something) \
    do { \
        int org_device_id = 0; \
        CUDA_CHECK(cudaGetDevice(&org_device_id)); \
        CUDA_CHECK(cudaSetDevice(device_id)); \
        something; \
        CUDA_CHECK(cudaSetDevice(org_device_id)); \
    } while (false)

/**
 * Do something on multiple devices, then switch back to the original device
 *
 *
 * example:
 *
 * DO_ON_MULTI_DEVICES(n_devices, [&](int device_id){
 *   //do_something_on_device(device_id);
 * });
 */

template<typename L>
void DO_ON_MULTI_DEVICES(int n_devices, L do_something) {
    int org_device_id = 0;
    CUDA_CHECK(cudaGetDevice(&org_device_id));
#pragma omp parallel for num_threads(n_devices)
    for (int device_id = 0; device_id < n_devices; device_id++) {
        CUDA_CHECK(cudaSetDevice(device_id));
        do_something(device_id);
    }
    CUDA_CHECK(cudaSetDevice(org_device_id));

}

#endif //FEDTREE_MULTI_DEVICE_H
