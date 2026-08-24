# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Best-effort shape tracing for NPU Triton kernel launches."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from vllm_ascend import envs


def dump_triton_kernel_shapes(
    kernel: str,
    grid: tuple[int, ...],
    arguments: Mapping[str, Any],
    launch_config: Mapping[str, Any],
) -> None:
    """Append tensor metadata and launch configuration for one Triton launch."""
    serialized_args: dict[str, Any] = {}
    for name, value in arguments.items():
        if isinstance(value, torch.Tensor):
            serialized_args[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        elif value is None:
            serialized_args[name] = None
        else:
            serialized_args[name] = value

    record = {
        "kernel": kernel,
        "grid": grid,
        "launch_config": launch_config,
        "arguments": serialized_args,
    }
    try:
        with Path(envs.VLLM_ASCEND_TRITON_SHAPE_DUMP_PATH).open(
            "a", encoding="utf-8"
        ) as file:
            file.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass
