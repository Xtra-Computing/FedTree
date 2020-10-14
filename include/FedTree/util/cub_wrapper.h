//
// Created by liqinbin on 10/14/20.
// ThunderGBM cub_wrapper.h: https://github.com/Xtra-Computing/thundergbm/blob/master/include/thundergbm/util/cub_wrapper.h
// Under Apache-2.0 license
// copyright (c) 2020 jiashuai
//

#ifndef FEDTREE_CUB_WRAPPER_H
#define FEDTREE_CUB_WRAPPER_H

#include <FedTree/syncarray.h>
#include "cub/cub.cuh"

template<typename T1, typename T2>
void cub_sort_by_key(SyncArray<T1> &keys, SyncArray<T2> &values, int size = -1, bool ascending = true,
                     void *temp = nullptr) {
    CHECK_EQ(values.size(), values.size()) << "keys and values must have equal size";
    using namespace cub;
    size_t num_items;
    if (-1 == size)
        num_items = keys.size();
    else
        num_items = size;
    SyncArray<char> temp_storage;
    DoubleBuffer<T1> d_keys;
    DoubleBuffer<T2> d_values;
    if (!temp) {
        SyncArray<T1> keys2(num_items);
        SyncArray<T2> values2(num_items);

        d_keys = DoubleBuffer<T1>(keys.device_data(), keys2.device_data());
        d_values = DoubleBuffer<T2>(values.device_data(), values2.device_data());
    } else {
        d_keys = DoubleBuffer<T1>(keys.device_data(), (T1 *) temp);
        d_values = DoubleBuffer<T2>(values.device_data(), (T2 *) ((T1 *) temp + num_items));
    }

    size_t temp_storage_bytes = 0;

    // Initialize device arrays
    if (ascending)
        DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_keys, d_values, num_items);
    else
        DeviceRadixSort::SortPairsDescending(NULL, temp_storage_bytes, d_keys, d_values, num_items);
    temp_storage.resize(temp_storage_bytes);

    // Run
    if (ascending)
        DeviceRadixSort::SortPairs(temp_storage.device_data(), temp_storage_bytes, d_keys, d_values, num_items);
    else
        DeviceRadixSort::SortPairsDescending(temp_storage.device_data(), temp_storage_bytes, d_keys, d_values,
                                             num_items);

    CUDA_CHECK(
            cudaMemcpy(keys.device_data(), reinterpret_cast<const void *>(d_keys.Current()), sizeof(T1) * num_items,
                       cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(values.device_data(), reinterpret_cast<const void *>(d_values.Current()),
                          sizeof(T2) * num_items,
                          cudaMemcpyDeviceToDevice));
}

template<typename T1, typename T2>
void cub_seg_sort_by_key(SyncArray<T1> &keys, SyncArray<T2> &values, const SyncArray<int> &ptr, bool ascending = true) {
    CHECK_EQ(values.size(), values.size()) << "keys and values must have equal size";
    using namespace cub;
    size_t num_items = keys.size();
    size_t num_segments = ptr.size() - 1;
    SyncArray<T1> keys2(num_items);
    SyncArray<T2> values2(num_items);
    SyncArray<char> temp_storage;

    DoubleBuffer<T1> d_keys(keys.device_data(), keys2.device_data());
    DoubleBuffer<T2> d_values(values.device_data(), values2.device_data());

    size_t temp_storage_bytes = 0;

    // Initialize device arrays
    if (ascending)
        DeviceSegmentedRadixSort::SortPairs(NULL, temp_storage_bytes, d_keys, d_values, num_items, num_segments,
                                            ptr.device_data(), ptr.device_data() + 1);
    else
        DeviceSegmentedRadixSort::SortPairsDescending(NULL, temp_storage_bytes, d_keys, d_values, num_items,
                                                      num_segments,
                                                      ptr.device_data(), ptr.device_data() + 1);
    temp_storage.resize(temp_storage_bytes);

    // Run
    if (ascending)
        DeviceSegmentedRadixSort::SortPairs(temp_storage.device_data(), temp_storage_bytes, d_keys, d_values,
                                            num_items, num_segments, ptr.device_data(),
                                            ptr.device_data() + 1);
    else
        DeviceSegmentedRadixSort::SortPairsDescending(temp_storage.device_data(), temp_storage_bytes, d_keys, d_values,
                                                      num_items, num_segments, ptr.device_data(),
                                                      ptr.device_data() + 1);

    CUDA_CHECK(
            cudaMemcpy(keys.device_data(), reinterpret_cast<const void *>(d_keys.Current()), sizeof(T1) * num_items,
                       cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(values.device_data(), reinterpret_cast<const void *>(d_values.Current()),
                          sizeof(T2) * num_items,
                          cudaMemcpyDeviceToDevice));
};



#endif //FEDTREE_CUB_WRAPPER_H
