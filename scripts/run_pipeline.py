"""Run resumable cloud stages without importing model frameworks in the controller."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import runpy
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from osc.collect.datasets import load_gsm8k, load_math500, save_jsonl
from osc.collect.pipeline import CollectConfig

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
STAGES = ("collect", "headroom", "elicit", "flips")
CONFIG_FLAGS = (
    "n_forced",
    "k_siblings",
    "sibling_stride",
    "sibling_max_tokens",
    "max_thinking_tokens",
    "max_steps",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workdir", type=Path)
    result.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    result.add_argument("--dataset", choices=("math500", "gsm8k"), default="math500")
    result.add_argument("--limit", type=int)
    result.add_argument("--max-hours", type=float, default=11)
    result.add_argument("--stages", default=",".join(STAGES))
    result.add_argument("--lam", type=float, nargs="+", default=[0.0, 0.05, 0.1])
    result.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--force", action="store_true")
    defaults = CollectConfig()
    for name in CONFIG_FLAGS:
        result.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(defaults, name)
        )
    return result


def stage_outputs(stage: str, args: argparse.Namespace) -> list[Path]:
    if stage == "collect":
        return [
            args.workdir / "traces" / "merged" / name
            for name in ("metadata.json", "traces.jsonl", "arrays.npz")
        ]
    if stage == "headroom":
        return [args.workdir / "results" / f"report_lam{lam:g}.json" for lam in args.lam]
    return [args.workdir / "results" / f"pilot_{stage}.json"]


def run_stage(stage: str, args: argparse.Namespace, remaining: float) -> None:
    """Use a child process so a scoring call cannot overrun the total budget."""
    traces = str(args.workdir / "traces" / "merged")
    if stage == "collect":
        command = [
            sys.executable,
            str(ROOT / "scripts" / "collect.py"),
            "--model",
            args.model,
            "--dataset",
            "jsonl",
            "--path",
            str(args.workdir / "problems.jsonl"),
            "--out",
            str(args.workdir / "traces"),
            "--dtype",
            args.dtype,
            "--attn-implementation",
            "sdpa",
            "--seed",
            str(args.seed),
            "--max-hours",
            str(max(remaining - 5, remaining * 0.9) / 3600),
        ]
        for name in CONFIG_FLAGS:
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        if args.force:
            command.append("--no-resume")
    elif stage == "headroom":
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--headroom-worker",
            "--traces",
            traces,
            "--out",
            str(args.workdir / "results"),
            "--seed",
            str(args.seed),
            "--lam",
            *map(str, args.lam),
        ]
    else:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--pilot-worker",
            stage,
            args.dtype,
            str(args.seed),
            "--traces",
            traces,
            "--problems",
            str(args.workdir / "problems.jsonl"),
            "--model",
            args.model,
            "--out",
            str(stage_outputs(stage, args)[0]),
        ]
    subprocess.run(command, check=True, timeout=remaining)


def pilot_worker(argv: list[str]) -> None:
    """Adapt legacy pilot entry points in the child, retaining their serialization."""
    from osc.collect import hf_backend

    stage, dtype, seed, *flags = argv
    original = hf_backend.HFBackend

    def backend(model_name: str):
        return original(model_name, dtype=dtype, seed=int(seed), attn_implementation="eager")

    hf_backend.HFBackend = backend
    sys.argv = [f"pilot_{stage}.py", *flags]
    runpy.run_path(str(ROOT / "scripts" / f"pilot_{stage}.py"), run_name="__main__")


def write_summary(args: argparse.Namespace, records: list[tuple[str, str, float]]) -> None:
    lines = [
        "# Cloud pipeline summary",
        "",
        f"Model: {args.model}",
        f"Dataset: {args.dataset}; seed: {args.seed}; dtype: {args.dtype}",
        "",
        "| Stage | Status | Seconds |",
        "| --- | --- | ---: |",
    ]
    lines.extend(f"| {stage} | {status} | {elapsed:.2f} |" for stage, status, elapsed in records)
    for path in sorted((args.workdir / "results").glob("*.json")):
        lines.extend(["", f"## {path.name}", ""])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key, value in payload.items():
                if value is None or isinstance(value, (str, int, float, bool)):
                    lines.append(f"- {key}: {value}")
        except (ValueError, OSError, AttributeError) as exc:
            lines.append(f"Could not read report: {exc}")
    summary = args.workdir / "SUMMARY.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with zipfile.ZipFile(args.workdir / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(summary, "SUMMARY.md")
        for path in sorted((args.workdir / "results").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(args.workdir))


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        cli.error("--max-hours must be finite and positive")
    if args.limit is not None and args.limit < 1:
        cli.error("--limit must be positive")
    if any(not math.isfinite(lam) or lam < 0 for lam in args.lam):
        cli.error("--lam values must be finite and nonnegative")
    requested = args.stages.split(",")
    if any(stage not in STAGES for stage in requested) or len(set(requested)) != len(requested):
        cli.error("--stages must contain unique names from " + ",".join(STAGES))
    try:
        CollectConfig(**{name: getattr(args, name) for name in CONFIG_FLAGS})
    except ValueError as exc:
        cli.error(str(exc))
    args.workdir = (
        args.workdir
        or Path("runs") / (re.sub(r"[^A-Za-z0-9._-]+", "-", args.model) + "-" + args.dataset)
    ).resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "results").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    deadline = time.monotonic() + args.max_hours * 3600
    records = []
    failed = False
    try:
        problems_path = args.workdir / "problems.jsonl"
        if args.force or not problems_path.exists():
            loader = load_math500 if args.dataset == "math500" else load_gsm8k
            save_jsonl(loader()[: args.limit], problems_path)
        for stage in STAGES:
            if stage not in requested:
                continue
            started = time.monotonic()
            remaining = deadline - started
            if not args.force and all(path.exists() for path in stage_outputs(stage, args)):
                status = "skipped (output exists)"
            elif remaining <= 0:
                status = "skipped (budget exhausted)"
                failed = True
            else:
                LOGGER.info("Starting %s with %.1f seconds remaining", stage, remaining)
                try:
                    run_stage(stage, args, remaining)
                    if not all(path.exists() for path in stage_outputs(stage, args)):
                        raise ValueError("stage returned without expected outputs")
                    status = "completed"
                except Exception as exc:
                    LOGGER.exception("Stage %s failed", stage)
                    status = f"failed: {type(exc).__name__}: {str(exc).replace('|', '/')}"
                    failed = True
            records.append((stage, status, time.monotonic() - started))
    finally:
        write_summary(args, records)
    return int(failed)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pilot-worker":
        pilot_worker(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "--headroom-worker":
        entry = runpy.run_path(str(ROOT / "scripts" / "headroom.py"))
        raise SystemExit(entry["main"](sys.argv[2:]))
    else:
        raise SystemExit(main())
