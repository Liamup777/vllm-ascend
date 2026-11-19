/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Adapted from CANN ops-transformer 66881f9a79f83f0e1c44902fe2c92eaceb26aecf.
// Keep arithmetic in sync with moe/moe_gating_top_k/op_kernel/arch35.
/*!
 * \file moe_gating_top_k_arch35_common.h
 * \brief
 */
#ifndef VLLM_ORDINARY_MOE_GATING_TOP_K_ARCH35_COMMON_H
#define VLLM_ORDINARY_MOE_GATING_TOP_K_ARCH35_COMMON_H

#include "kernel_operator.h"
#include "load_store_utils.h"
#include "basic_api/kernel_operator_utils_intf.h"

namespace MoeGatingTopKOrdinary {
using namespace AscendC;
using Reg::RegTensor;
constexpr int32_t MIN_FP32 = 0xFF800000;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t BLOCK_BYTES = 32;
constexpr uint32_t VL_FLOAT_SIZE = VECTOR_REG_WIDTH / sizeof(float);

constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;

constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;

// softplus 稳定公式中 log1p(t) 的泰勒多项式切换阈值(2^-8, 1/256, fp32 精确表示):
// t < 该值时用 t - t^2/2 近似(截断相对误差 t^2/3 <= 2^-17),
// t >= 该值时直接 ln(1+t)(舍入相对误差 ~2^-24/t <= 2^-16), 均远小于比对阈值 2^-13
constexpr float LOG1P_TAYLOR_THRESHOLD = 0.00390625f;

constexpr int64_t NORM_TYPE_SOFTMAX = 0;
constexpr int64_t NORM_TYPE_SIGMOID = 1;
constexpr int64_t NORM_TYPE_SOFTPLUS = 2;
constexpr int64_t NORM_TYPE_SIGMOID_OPTIONAL_RENORM = 3;

__aicore__ inline bool IsSigmoidNormType(int64_t normType)
{
    return normType == NORM_TYPE_SIGMOID || normType == NORM_TYPE_SIGMOID_OPTIONAL_RENORM;
}

// Keep the legacy behavior for normType 0, 1 and 2. The new normType 3
// selects Sigmoid with renormalization controlled by renorm.
__aicore__ inline bool ShouldRenormalize(int64_t normType, int64_t renorm)
{
    return normType == NORM_TYPE_SIGMOID || normType == NORM_TYPE_SOFTPLUS ||
           ((normType == NORM_TYPE_SOFTMAX || normType == NORM_TYPE_SIGMOID_OPTIONAL_RENORM) && renorm == 1);
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a < b ? b : a;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 x, T2 y)
{
    if (y != 0 && x != 0) {
        const T1 quotient = x / y;
        return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
    }

    return x;
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

} // namespace MoeGatingTopKOrdinary

namespace MoeGatingTopKOrdinary {

template <typename T>
__aicore__ inline void SmallKAlignEVFNoNorm(__ubuf__ float* inputAddr, __ubuf__ uint32_t* mrgSortAddr,
                                            __ubuf__ T* outputAddr, __ubuf__ uint32_t* expertIdxAddr, uint32_t k,
                                            float routedScalingFactor)
{
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> vregSortValue;
        RegTensor<uint32_t> vregExpertIdx;
        RegTensor<float> vregGathered;

        Reg::MaskReg preg0 = Reg::UpdateMask<float>(k);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B32>(vregSortValue, vregExpertIdx, mrgSortAddr);
        Reg::Gather(vregGathered, inputAddr, vregExpertIdx, preg0);
        Reg::Muls(vregGathered, vregGathered, routedScalingFactor, preg0);
        MoeGatingTopKOrdinaryIO::StoreOneTensorForDtypeT<T>(outputAddr, vregGathered, preg0, 0);
        Reg::StoreAlign(expertIdxAddr, vregExpertIdx, preg0);
    }
}

template <typename T>
__aicore__ inline void SmallKAlignEVFWithNorm(__ubuf__ float* inputAddr, __ubuf__ uint32_t* mrgSortAddr,
                                              __ubuf__ T* outputAddr, __ubuf__ uint32_t* expertIdxAddr, uint32_t k,
                                              float eps, float routedScalingFactor)
{
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> vregSortValue;
        RegTensor<uint32_t> vregExpertIdx;
        RegTensor<float> vregGathered;
        RegTensor<float> vregSum;
        RegTensor<float> vregSumBcast;

        Reg::MaskReg preg0 = Reg::UpdateMask<float>(k);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B32>(vregSortValue, vregExpertIdx, mrgSortAddr);
        Reg::Gather(vregGathered, inputAddr, vregExpertIdx, preg0);
        Reg::Reduce<Reg::ReduceType::SUM>(vregSum, vregGathered, preg0);
        Reg::Adds(vregSum, vregSum, eps, preg0);
        Reg::Duplicate(vregSumBcast, vregSum, preg0);
        Reg::Div(vregSumBcast, vregGathered, vregSumBcast, preg0);
        Reg::Muls(vregSumBcast, vregSumBcast, routedScalingFactor, preg0);
        MoeGatingTopKOrdinaryIO::StoreOneTensorForDtypeT<T>(outputAddr, vregSumBcast, preg0, 0);
        Reg::StoreAlign(expertIdxAddr, vregExpertIdx, preg0);
    }
}

template <typename T>
__aicore__ inline void LargeKAlignEVFNoNorm(__ubuf__ float* inputAddr, __ubuf__ uint32_t* mrgSortAddr,
                                            __ubuf__ T* outputAddr, __ubuf__ uint32_t* expertIdxAddr, uint32_t k,
                                            float routedScalingFactor)
{
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> vregSortValue;
        RegTensor<uint32_t> vregExpertIdx;
        RegTensor<float> vregGathered;
        Reg::MaskReg preg0 = Reg::CreateMask<float>();

        uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(k, VL_FLOAT_SIZE));
        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<uint32_t>(k);
            Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B32>(vregSortValue, vregExpertIdx,
                                                                     mrgSortAddr + i * 2 * VL_FLOAT_SIZE);
            Reg::Gather(vregGathered, inputAddr, vregExpertIdx, preg0);
            Reg::Muls(vregGathered, vregGathered, routedScalingFactor, preg0);
            MoeGatingTopKOrdinaryIO::StoreOneTensorForDtypeT<T>(outputAddr, vregGathered, preg0, i * VL_FLOAT_SIZE);
            Reg::StoreAlign(expertIdxAddr + i * VL_FLOAT_SIZE, vregExpertIdx, preg0);
        }
    }
}

template <typename T>
__aicore__ inline void LargeKAlignEVFWithNorm(__ubuf__ float* inputAddr, __ubuf__ uint32_t* mrgSortAddr,
                                              __ubuf__ T* outputAddr, __ubuf__ uint32_t* expertIdxAddr, uint32_t k,
                                              float eps, float routedScalingFactor)
{
    uint32_t k1 = k;
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> vregSortValue;
        RegTensor<uint32_t> vregExpertIdx;
        RegTensor<float> vregGathered;
        RegTensor<float> vregSumBcast;
        RegTensor<float> vregOutput;
        RegTensor<float> vregSum;

        Reg::MaskReg preg0 = Reg::CreateMask<float>();
        Reg::MaskReg preg1 = Reg::CreateMask<float>();
        Reg::Duplicate(vregSum, static_cast<float>(0), preg0);
        uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(k, VL_FLOAT_SIZE));

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = Reg::UpdateMask<uint32_t>(k);
            Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B32>(vregSortValue, vregExpertIdx,
                                                                     mrgSortAddr + i * 2 * VL_FLOAT_SIZE);
            Reg::Duplicate(vregGathered, static_cast<float>(0), preg1);
            Reg::Gather(vregGathered, inputAddr, vregExpertIdx, preg0);
            Reg::Add(vregSum, vregSum, vregGathered, preg1);
        }

        Reg::Reduce<Reg::ReduceType::SUM>(vregSum, vregSum, preg1);
        Reg::Adds(vregSum, vregSum, eps, preg1);
        Reg::Duplicate(vregSumBcast, vregSum, preg1);
        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg1 = Reg::UpdateMask<uint32_t>(k1);
            Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B32>(vregSortValue, vregExpertIdx,
                                                                     mrgSortAddr + i * 2 * VL_FLOAT_SIZE);
            Reg::Gather(vregGathered, inputAddr, vregExpertIdx, preg1);
            Reg::Div(vregOutput, vregGathered, vregSumBcast, preg1);
            Reg::Muls(vregOutput, vregOutput, routedScalingFactor, preg1);
            MoeGatingTopKOrdinaryIO::StoreOneTensorForDtypeT<T>(outputAddr, vregOutput, preg1, i * VL_FLOAT_SIZE);
            Reg::StoreAlign(expertIdxAddr + i * VL_FLOAT_SIZE, vregExpertIdx, preg1);
        }
    }
}

} // namespace MoeGatingTopKOrdinary
#endif // VLLM_ORDINARY_MOE_GATING_TOP_K_ARCH35_COMMON_H
