# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend compatibility kernel for sampling-mask packing."""

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors

_BLOCK_SIZE = 4096


@triton.jit
def _pack_sampling_mask_kernel_ascend(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    num_sampled_tokens_ptr,
    packed_mask_ptr,
    packed_mask_row_stride,
    counts_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_active = tl.load(num_sampled_tokens_ptr + req_idx) > 0
    count = tl.zeros((), dtype=tl.int32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        valid = offsets < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_row_stride + offsets * logits_col_stride,
            mask=valid,
            other=-float("inf"),
        )
        keep = (logits > -float("inf")) & (logits < float("inf")) & is_active
        # Triton-Ascend reduces an i1 tensor as a boolean. Cast before the sum
        # so counts contains the number of finite logits, not the tile count.
        count += tl.sum(keep.to(tl.int32))

        keep = tl.trans(tl.reshape(keep.to(tl.int32), (BLOCK_SIZE // 8, 8)))
        bit_weights = (1 << tl.arange(0, 8))[:, None]
        packed = tl.sum(keep * bit_weights, axis=0).to(tl.uint8)
        byte_offsets = start_idx // 8 + tl.arange(0, BLOCK_SIZE // 8)
        tl.store(
            packed_mask_ptr + req_idx * packed_mask_row_stride + byte_offsets,
            packed,
            mask=byte_offsets < tl.cdiv(vocab_size, 8),
        )

    tl.store(counts_ptr + req_idx, count)


def sampling_mask_from_logits_npu(
    cls: type[SamplingMaskTensors],
    logits: torch.Tensor,
    num_sampled_tokens: torch.Tensor,
) -> SamplingMaskTensors:
    """Pack sampling masks after making strided vocab loads NPU-friendly."""
    if logits.stride(1) != 1:
        logits = logits.contiguous()

    num_reqs, vocab_size = logits.shape
    packed_width = (vocab_size + 7) // 8
    packed_mask = torch.empty(
        (num_reqs, packed_width),
        dtype=torch.uint8,
        device=logits.device,
    )
    counts = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
    _pack_sampling_mask_kernel_ascend[(num_reqs,)](
        logits,
        logits.stride(0),
        logits.stride(1),
        num_sampled_tokens,
        packed_mask,
        packed_mask.stride(0),
        counts,
        vocab_size,
        BLOCK_SIZE=_BLOCK_SIZE,
    )

    return cls(packed_mask, counts, vocab_size)
