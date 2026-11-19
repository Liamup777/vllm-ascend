// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#ifndef VLLM_GATING_ORDINARY_LOAD_STORE_UTILS_H
#define VLLM_GATING_ORDINARY_LOAD_STORE_UTILS_H

#include "kernel_operator.h"

// Local dtype adapter for the CANN ordinary kernels. Preserve FP32 loads and
// round-to-nearest output casts without depending on ops-transformer's build.
namespace MoeGatingTopKOrdinaryIO {
using namespace AscendC;
constexpr Reg::CastTrait widen = {
    Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr Reg::CastTrait narrow = {
    Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T>
__aicore__ inline void LoadOneTensorForDtypeT(__local_mem__ T* input, Reg::RegTensor<float>& dst,
                                            Reg::MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        Reg::LoadAlign(dst, input + offset);
    } else {
        Reg::RegTensor<T> value;
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(value, input + offset);
        Reg::Cast<float, T, widen>(dst, value, mask);
    }
}

template <typename T>
__aicore__ inline void LoadTwoTensorForDtypeT(__local_mem__ T* first, __local_mem__ T* second,
                                            Reg::RegTensor<float>& firstValue, Reg::RegTensor<float>& secondValue,
                                            Reg::MaskReg& firstMask, Reg::MaskReg& secondMask,
                                            uint32_t firstOffset, uint32_t secondOffset)
{
    LoadOneTensorForDtypeT(first, firstValue, firstMask, firstOffset);
    LoadOneTensorForDtypeT(second, secondValue, secondMask, secondOffset);
}

template <typename T>
__aicore__ inline void StoreOneTensorForDtypeT(__local_mem__ T* output, Reg::RegTensor<float>& value,
                                             Reg::MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        Reg::StoreAlign(output + offset, value, mask);
    } else {
        Reg::RegTensor<T> converted;
        Reg::Cast<T, float, narrow>(converted, value, mask);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(output + offset, converted, mask);
    }
}
} // namespace MoeGatingTopKOrdinaryIO
#endif
