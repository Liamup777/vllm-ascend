#!/usr/bin/env python3
"""Run one explicit MRv2 offline coverage scenario."""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


def expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand(x) for x in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_scenes(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    raw = expand(json.loads(path.read_text(encoding="utf-8")))
    defaults = raw["defaults"]
    scenes = {x["id"]: merge(defaults, x) for x in raw["scenarios"]}
    return defaults, scenes


def describe(scene: dict[str, Any]) -> None:
    print(f"{scene['id']}  {scene['name']}")
    print(f"测试目标: {scene['purpose']}")
    print("覆盖算子:")
    for op in scene["expected_operators"]:
        print(f"  - {op}")
    print(f"模型: {scene['model']}")
    print(f"设备: {scene['device']}")
    print(f"LLM 参数: {json.dumps(scene['engine'], ensure_ascii=False)}")
    print("测试 case:")
    for case in scene["cases"]:
        sampling_base = (
            {} if scene.get("replace_hot_sampling") else scene.get("hot_sampling", {})
        )
        sampling = merge(sampling_base, case["sampling"])
        print(
            f"  - {case['name']}: count={case['count']}, "
            f"prompt_tokens≈{case['prompt_tokens']}, "
            f"SamplingParams={json.dumps(sampling, ensure_ascii=False)}"
        )
    print(
        f"采集: warmup={scene['warmup_rounds']} rounds, "
        f"profile={scene['profile_rounds']} rounds"
    )
    if scene.get("skip"):
        print("状态: skip=true；填好模型/设备后改为 false")


def make_prompt(tokenizer: Any, scene: dict[str, Any], req: dict[str, Any]) -> str:
    if messages := req.get("messages"):
        enable_thinking = req.get(
            "enable_thinking", scene.get("enable_thinking", False)
        )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    target = int(req.get("prompt_tokens", 0))
    text = req["prompt_text"]
    if target <= 0:
        return text
    unit_ids = tokenizer.encode(text, add_special_tokens=False)
    repeats = max(1, target // max(1, len(unit_ids)) + 1)
    ids = (unit_ids * repeats)[:target]
    return tokenizer.decode(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("scenarios.json"))
    parser.add_argument("--scene")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--describe")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile", choices=("torch", "none"), default="torch")
    args = parser.parse_args()

    _, scenes = load_scenes(args.config)
    if args.list:
        for scene in scenes.values():
            print(f"{scene['id']:>3}  {scene['name']:<28} skip={scene.get('skip', False)}")
        return
    scene_id = args.describe or args.scene
    if not scene_id or scene_id not in scenes:
        raise SystemExit(f"Use --scene/--describe; choices: {','.join(scenes)}")
    scene = scenes[scene_id]
    describe(scene)
    if args.describe or args.dry_run:
        return
    if scene.get("skip"):
        raise SystemExit("This scenario is skip=true; configure its model/device first.")
    if "${" in str(scene["model"]):
        raise SystemExit("Model environment variable is unresolved.")

    env = {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "ASCEND_RT_VISIBLE_DEVICES": str(scene["device"]),
    }
    # ModelScope treats a local absolute path as a remote repository when it
    # is enabled. Keep it only for model identifiers that are not local dirs.
    env["VLLM_USE_MODELSCOPE"] = str(
        not Path(scene["model"]).expanduser().is_dir()
    )
    env.update(scene.get("environment", {}))
    os.environ.update({str(k): str(v) for k, v in env.items()})

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RepetitionDetectionParams, StructuredOutputsParams

    engine = deepcopy(scene["engine"])
    profile_dir = Path(scene["profile_root"]) / scene["id"]
    if args.profile == "torch":
        profile_dir.mkdir(parents=True, exist_ok=True)
        engine["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir),
            "torch_profiler_with_stack": False,
        }
    llm = LLM(model=scene["model"], **engine)
    tokenizer = AutoTokenizer.from_pretrained(scene["model"], trust_remote_code=True)
    prepared_cases: list[tuple[str, list[Any], Any]] = []
    image = None
    if scene.get("image_path"):
        from PIL import Image
        image = Image.open(scene["image_path"]).convert("RGB")
    for case in scene["cases"]:
        prompt = make_prompt(tokenizer, scene, case)
        prompts: list[Any] = [prompt] * int(case["count"])
        if image is not None:
            prompts = [{"prompt": prompt, "multi_modal_data": {"image": image}}] * int(case["count"])
        sampling_base = (
            {} if scene.get("replace_hot_sampling") else scene.get("hot_sampling", {})
        )
        sampling_kwargs = merge(sampling_base, case["sampling"])
        if "logit_bias" in sampling_kwargs:
            sampling_kwargs["logit_bias"] = {
                int(token_id): bias
                for token_id, bias in sampling_kwargs["logit_bias"].items()
            }
        if "structured_outputs" in sampling_kwargs:
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(**sampling_kwargs["structured_outputs"])
        if "repetition_detection" in sampling_kwargs:
            sampling_kwargs["repetition_detection"] = RepetitionDetectionParams(**sampling_kwargs["repetition_detection"])
        prepared_cases.append((case["name"], prompts, SamplingParams(**sampling_kwargs)))

    for index in range(int(scene["warmup_rounds"])):
        for case_name, prompts, params in prepared_cases:
            llm.generate(prompts, params)
            print(f"WARMUP_DONE round={index + 1} case={case_name}", flush=True)
    print("PROFILE_REGION_BEGIN", flush=True)
    if args.profile == "torch":
        llm.start_profile()
    try:
        for index in range(int(scene["profile_rounds"])):
            for case_name, prompts, params in prepared_cases:
                outputs = llm.generate(prompts, params)
                print(
                    f"PROFILE_ROUND_DONE round={index + 1} "
                    f"case={case_name} outputs={len(outputs)}",
                    flush=True,
                )
    finally:
        if args.profile == "torch":
            llm.stop_profile()
    print("PROFILE_REGION_END", flush=True)


if __name__ == "__main__":
    main()
