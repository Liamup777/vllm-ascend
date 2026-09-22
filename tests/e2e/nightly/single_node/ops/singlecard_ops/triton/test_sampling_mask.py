# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for sampling-mask packing on non-contiguous logits."""

import torch
from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.v2.sample.sampling_mask import sampling_mask_from_logits_npu


def test_sampling_mask_non_contiguous_vocab_dimension():
    init_device_properties_triton()

    num_reqs = 4
    vocab_size = 128256
    base = torch.full(
        (num_reqs, vocab_size * 2),
        -float("inf"),
        dtype=torch.float32,
        device="npu",
    )
    logits = base[:, ::2]
    logits[:, :64] = torch.randn(
        (num_reqs, 64),
        dtype=logits.dtype,
        device=logits.device,
    )

    assert logits.shape == (num_reqs, vocab_size)
    assert logits.stride(1) == 2

    num_sampled = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device="npu")
    tensors = sampling_mask_from_logits_npu(SamplingMaskTensors, logits, num_sampled)

    # The wrapper rebinds only its local argument; the caller's view is unchanged.
    assert logits.stride(1) == 2
    torch.npu.synchronize()
    torch.testing.assert_close(
        tensors.counts,
        torch.tensor([64, 0, 64, 0], dtype=torch.int32, device="npu"),
        rtol=0,
        atol=0,
    )
    result = tensors.tolists(num_sampled.cpu().numpy())

    assert result.to_nested_list() == [list(range(64))] * 2
