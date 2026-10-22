"""Run the elicit pilot using an optional Hugging Face backend."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from osc.data.io import load_traceset
from osc.pilots.elicit import run_elicitation_pilot


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
    args = parser.parse_args()
    if args.every < 1:
        parser.error("--every must be positive")
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

    backend = HFBackend(args.model)
    result = run_elicitation_pilot(backend, traces, prompts, every=args.every)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_json_value(asdict(result)), indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
