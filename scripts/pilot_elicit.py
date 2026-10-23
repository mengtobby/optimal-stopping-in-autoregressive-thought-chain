"""Run the elicit pilot using an optional Hugging Face backend."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from osc.data.io import load_traceset
from osc.pilots.elicit import run_elicitation_pilot_detailed


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--problems", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="collection CollectConfig.max_steps (defaults to trace metadata)",
    )
    parser.add_argument("--include-censored", action="store_true")
    parser.add_argument("--method", choices=("distribution", "greedy"), default="distribution")
    parser.add_argument("--max-k", type=int, default=30)
    parser.add_argument("--min-mass", type=float, default=0.0)
    parser.add_argument(
        "--score-chunk-size", type=int, default=8, help="candidate batch size for KV-cache copies"
    )
    args = parser.parse_args()
    if not 0 <= args.max_k <= 10000:
        parser.error("--max-k must be in [0, 10000]")
    if not math.isfinite(args.min_mass) or not 0 <= args.min_mass <= 1:
        parser.error("--min-mass must be finite and in [0, 1]")
    if args.every < 1:
        parser.error("--every must be positive")
    if args.score_chunk_size < 1:
        parser.error("--score-chunk-size must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    prompts = {}
    with args.problems.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not all(
                isinstance(record.get(key), str) for key in ("problem_id", "prompt")
            ):
                parser.error(f"invalid problem record at line {line_number}")
            if record["problem_id"] in prompts:
                parser.error(f"duplicate problem_id at line {line_number}")
            prompts[record["problem_id"]] = record["prompt"]
    traces = load_traceset(args.traces)
    if any(trace.problem_id not in prompts for trace in traces):
        parser.error("problems file must contain a prompt for every trace")
    from osc.collect.hf_backend import HFBackend

    backend_options = (
        {} if args.score_chunk_size == 8 else {"score_chunk_size": args.score_chunk_size}
    )
    backend = HFBackend(args.model, **backend_options)
    result, details = run_elicitation_pilot_detailed(
        backend,
        traces,
        prompts,
        every=args.every,
        include_censored=args.include_censored,
        max_steps=args.max_steps,
        method=args.method,
        max_k=args.max_k,
        min_mass=args.min_mass,
    )
    payload = asdict(result)
    payload["per_trace"] = [
        {"expected": detail.expected, "mode": detail.mode, "mass": detail.mass}
        for detail in details
    ]
    masses = np.concatenate([detail.mass for detail in details]) if details else np.array([])
    finite_mass = masses[np.isfinite(masses)]
    payload["mean_mass"] = float(finite_mass.mean()) if finite_mass.size else float("nan")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_json_value(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
