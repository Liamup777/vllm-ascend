# Sampling-mask packing (`SamplingMaskTensors.from_logits`)

## Description

- **Function**: identifies finite logits for active requests, packs the boolean
  keep mask into bytes, and counts the kept vocabulary entries.
- **Python entry point**: `sampling_mask_from_logits_npu` in
  `vllm_ascend/ops/triton/v2/sample/sampling_mask.py`.
- **Triton kernel**: `_pack_sampling_mask_kernel_ascend`.
- **Integration**: `vllm_ascend/patch/worker/patch_v2/patch_triton.py` binds the
  wrapper directly as `SamplingMaskTensors.from_logits`.

For request row `r` and vocabulary index `i`, the keep condition is:

```text
keep[r, i] = isfinite(logits[r, i]) and num_sampled_tokens[r] > 0
```

The output bit layout is:

```text
packed_mask[r, i // 8] bit (i % 8) = keep[r, i]
counts[r] = sum_i keep[r, i]
```

An inactive request therefore has a zero mask and a zero count. Vocabulary
tails that are not divisible by eight are masked and do not contribute bits or
counts.

## Data flow

The kernel processes one request per Triton program and scans its vocabulary in
4096-element blocks:

```text
logits tile [4096]
    -> finite and active mask [4096]
    -> int32 sum -> row count
    -> reshape [512, 8]
    -> transpose [8, 512]
    -> multiply by bit weights [8, 1]
    -> reduce axis 0
    -> packed bytes [512]
```

The transpose makes the eight bits for 512 output bytes the reduction axis.
The bit weights are compile-time constants, so the packing lowers to vector
transpose, multiply, and reduction operations instead of a tensor-wide loop of
variable shifts.

The 4096-element block size also limits temporary storage. The original
8192-element implementation can exceed Ascend UB capacity for a large
non-contiguous vocabulary input.

## Parameters and outputs

| Name | Input/Output/Attribute | Shape | Data type | Description |
| --- | --- | --- | --- | --- |
| `logits` | Input | `[num_reqs, vocab_size]` | floating point | Per-request logits. NaN, `+inf`, and `-inf` entries are excluded. |
| `num_sampled_tokens` | Input | `[num_reqs]` | integer | A row is active only when its value is greater than zero. |
| `packed_mask` | Output | `[num_reqs, ceil(vocab_size / 8)]` | uint8 | Little-bit-order packed finite-logit mask. |
| `counts` | Output | `[num_reqs]` | int32 | Number of set bits in each row. |
| `vocab_size` | Output metadata | scalar | Python int | Original vocabulary size used when unpacking the mask. |

The wrapper returns `SamplingMaskTensors(packed_mask, counts, vocab_size)`.

## Stride handling

The Triton kernel accepts explicit row and column strides, but strided loads
along the vocabulary dimension can cause excessive UB allocation in the
Triton-Ascend compiler. The wrapper therefore creates a contiguous local copy
only when:

```text
logits.stride(1) != 1
```

Rebinding the local Python variable does not mutate or replace the caller's
tensor. The caller keeps the same storage, shape, and strides. Contiguous
inputs are passed through without a copy.

## Constraints

- `logits` is a rank-2 tensor and `num_sampled_tokens` has one entry per row.
- The operator is inference-only and has no backward path.
- The vocabulary size may be non-aligned to both 4096 elements and eight bits;
  masked tail lanes are supported.
- The finite-logit count is accumulated as int32. The boolean keep mask is
  explicitly converted before reduction because boolean reduction does not
  provide the required integer count on affected Triton-Ascend versions.
- The wrapper-level contiguous copy is separate from the Triton kernel launch;
  kernel-only profiling does not include that copy cost.
- The target vLLM dependency provides the three-field `SamplingMaskTensors`
  layout required by this implementation.

## Origin and differences

- **Origin**: adapts vLLM Model Runner V2
  `SamplingMaskTensors.from_logits` and its sampling-mask packing kernel for
  Ascend NPU.
- **Correctness adaptation**: casts the keep mask to int32 before summation so
  `counts` contains the number of finite logits.
- **Memory adaptation**: uses 4096-element tiles and makes only a strided
  vocabulary dimension contiguous before launch to avoid UB overflow.
- **Performance adaptation**: replaces per-element variable shifts and a
  scalar-heavy `[512, 8]` axis-1 reduction with an `[8, 512]` vector reduction
  using compile-time bit weights.
- The packed bit order, active-row semantics, count values, and returned
  `SamplingMaskTensors` contract are unchanged.

## Test cases

The NPU regression test uses a large stride-2 vocabulary view and covers active
and inactive request rows. It checks exact counts, unpacked token indices, and
that the caller's non-contiguous `logits` view is unchanged after the call.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sampling_mask.py
```

The test is registered as the A2 single-card nightly case `sampling-mask` in
`.github/workflows/configs/nightly_config.yaml`.
