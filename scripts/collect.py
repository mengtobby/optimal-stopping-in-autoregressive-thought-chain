"""Collect reasoning traces from a local or Hugging Face problem dataset."""

import argparse
import logging
import math
import time
from dataclasses import fields

from osc.collect.datasets import load_gsm8k, load_jsonl, load_math500
from osc.collect.hf_backend import HFBackend
from osc.collect.pipeline import CollectConfig, collect_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=["math500", "gsm8k", "jsonl"], required=True)
    parser.add_argument("--path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto"
    )
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    for field in fields(CollectConfig):
        parser.add_argument(
            "--" + field.name.replace("_", "-"), type=type(field.default), default=field.default
        )
    args = parser.parse_args()
    if args.dataset == "jsonl" and not args.path:
        parser.error("--path is required for jsonl")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_hours is not None and (not math.isfinite(args.max_hours) or args.max_hours <= 0):
        parser.error("--max-hours must be finite and positive")
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    deadline = None if args.max_hours is None else time.monotonic() + args.max_hours * 3600
    loaders = {"math500": load_math500, "gsm8k": load_gsm8k, "jsonl": load_jsonl}
    problems = loaders[args.dataset](args.path)
    config = CollectConfig(**{f.name: getattr(args, f.name) for f in fields(CollectConfig)})
    backend = HFBackend(
        args.model,
        device=args.device,
        seed=args.seed,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    collect_dataset(
        backend,
        problems[: args.limit],
        config,
        args.out,
        resume=not args.no_resume,
        deadline=deadline,
    )


if __name__ == "__main__":
    main()
