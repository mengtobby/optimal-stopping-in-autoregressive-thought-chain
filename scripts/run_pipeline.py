"""Run resumable cloud stages without importing model frameworks in the controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import runpy
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from osc.collect.datasets import load_gsm8k, load_jsonl, load_math500, save_jsonl
from osc.collect.pipeline import CollectConfig
from osc.data.io import load_traceset

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
    result.add_argument(
        "--collect-reserve-minutes",
        type=float,
        default=30,
        help="minimum time reserved after collection's soft deadline (default: 30 minutes)",
    )
    result.add_argument("--stages", default=",".join(STAGES))
    result.add_argument("--lam", type=float, nargs="+", default=[0.0, 0.05, 0.1])
    result.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--force", action="store_true")
    result.add_argument("--refresh-problems", action="store_true")
    result.add_argument("--score-chunk-size", type=int, default=8)
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


def problem_ids(args: argparse.Namespace) -> set[str]:
    """Read the expected IDs independently of collection and merged output."""
    path = args.workdir / "problems.jsonl"
    if not path.exists():
        return set()
    return {problem.problem_id for problem in load_jsonl(path)}


def collection_progress(args: argparse.Namespace) -> tuple[int, int]:
    """Count expected problem IDs with committed shards, excluding temporary shards."""
    ids = problem_ids(args)
    collected = 0
    for problem_id in ids:
        shard = args.workdir / "traces" / "shards" / hashlib.sha256(problem_id.encode()).hexdigest()
        if all(
            (shard / name).is_file() for name in ("metadata.json", "traces.jsonl", "arrays.npz")
        ):
            collected += 1
    return collected, len(ids)


def trace_count(args: argparse.Namespace) -> int:
    path = args.workdir / "traces" / "merged" / "traces.jsonl"
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def merged_fingerprint(workdir: Path) -> str | None:
    """Hash validated merged IDs and serialized content without loading archive bytes at once."""
    merged = workdir / "traces" / "merged"
    try:
        traces = load_traceset(merged)
        digest = hashlib.sha256()
        digest.update(json.dumps(sorted(trace.problem_id for trace in traces)).encode())
        for name in ("metadata.json", "traces.jsonl", "arrays.npz"):
            digest.update(name.encode())
            with (merged / name).open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        return digest.hexdigest()
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        EOFError,
        zipfile.BadZipFile,
    ):
        return None


def dated_path(path: Path, label: str) -> Path:
    """Choose a unique sibling name, including for multiple invocations in one second."""
    return path.with_name(f"{path.name}.{label}-{time.time_ns()}")


def prepare_collection(args: argparse.Namespace) -> Path | None:
    merged = args.workdir / "traces" / "merged"
    backup = None
    if merged_fingerprint(args.workdir) is not None:
        backup = dated_path(merged, "bak")
        shutil.copytree(merged, backup)
        LOGGER.warning("Preserving existing merged traces before collection in %s", backup)
    shards = args.workdir / "traces" / "shards"
    if args.force and shards.exists():
        old = dated_path(shards, "old")
        shards.rename(old)
        LOGGER.warning("Forced collection moved previous shards to %s", old)
    return backup


def restore_collection(args: argparse.Namespace, backup: Path | None) -> str:
    if backup is None:
        return ""
    merged = args.workdir / "traces" / "merged"
    old_count = len(load_traceset(backup))
    if merged_fingerprint(args.workdir) is not None and len(load_traceset(merged)) >= old_count:
        return ""
    partial = dated_path(merged, "partial")
    if merged.exists():
        merged.rename(partial)
    shutil.copytree(backup, merged)
    message = f"restored {old_count} problems from {backup.name}; partial output: {partial.name}"
    LOGGER.warning(message)
    return message


def provenance_path(output: Path) -> Path:
    return output.with_suffix(".resume.json")


def stage_complete(stage: str, args: argparse.Namespace) -> bool:
    outputs = stage_outputs(stage, args)
    if not all(path.is_file() for path in outputs):
        return False
    if stage == "collect":
        collected, total = collection_progress(args)
        if collected != total:
            return False
        try:
            # Loading validates JSON, array keys/shapes, and the NPZ archive. A kill
            # during save can leave new JSON beside an old or truncated archive.
            merged = load_traceset(args.workdir / "traces" / "merged")
            return len(merged) == total and {trace.problem_id for trace in merged} == problem_ids(
                args
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            IndexError,
            EOFError,
            zipfile.BadZipFile,
        ):
            LOGGER.warning("Merged traces are incomplete or invalid; re-merging from shards")
            return False
    fingerprint = merged_fingerprint(args.workdir)
    if fingerprint is None:
        return False
    for output in outputs:
        try:
            if (
                json.loads(provenance_path(output).read_text(encoding="utf-8")).get("fingerprint")
                != fingerprint
            ):
                return False
        except (OSError, ValueError, AttributeError):
            return False
    return True


def run_stage(stage: str, args: argparse.Namespace, remaining: float) -> None:
    """Use a child process so a scoring call cannot overrun the total budget."""
    traces = str(args.workdir / "traces" / "merged")
    if stage == "collect":
        # Stop starting problems well before the hard timeout: the last problem
        # must finish and shards must merge before downstream stages can run.
        reserve = max(args.collect_reserve_minutes * 60, remaining * 0.25)
        soft_seconds = max(min(1.0, remaining * 0.1), remaining - reserve)
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
            str(soft_seconds / 3600),
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
    if stage == "elicit":
        command.extend(["--score-chunk-size", str(args.score_chunk_size)])
    subprocess.run(command, check=True, timeout=remaining)


def pilot_worker(argv: list[str]) -> None:
    """Adapt legacy pilot entry points in the child, retaining their serialization."""
    from osc.collect import hf_backend

    stage, dtype, seed, *flags = argv
    original = hf_backend.HFBackend

    def backend(*args, **kwargs):
        defaults = {
            "dtype": dtype,
            "seed": int(seed),
            "attn_implementation": "eager" if stage == "flips" else "sdpa",
        }
        # Positional constructor options take precedence over injected defaults too.
        for name in (
            "device",
            "dtype",
            "feature_layers",
            "temperature",
            "top_p",
            "seed",
            "score_chunk_size",
            "attn_implementation",
        )[: max(0, len(args) - 1)]:
            defaults.pop(name, None)
        return original(*args, **{**defaults, **kwargs})

    hf_backend.HFBackend = backend
    sys.argv = [f"pilot_{stage}.py", *flags]
    runpy.run_path(str(ROOT / "scripts" / f"pilot_{stage}.py"), run_name="__main__")


def write_summary(args: argparse.Namespace, records: list[tuple[str, str, float]]) -> None:
    try:
        collected, total = collection_progress(args)
        progress = f"Collected {collected} of {total} problems."
    except (OSError, ValueError, TypeError) as exc:
        progress = f"Could not read problems.jsonl: {exc}"
    lines = [
        "# Cloud pipeline summary",
        "",
        f"Model: {args.model}",
        f"Dataset: {args.dataset}; seed: {args.seed}; dtype: {args.dtype}",
        "",
        progress,
        "Resume by rerunning the same command / notebook with this workdir.",
        "",
        "| Stage | Status | Seconds |",
        "| --- | --- | ---: |",
    ]
    lines.extend(f"| {stage} | {status} | {elapsed:.2f} |" for stage, status, elapsed in records)
    for path in sorted((args.workdir / "results").glob("*.json")):
        if path.name.endswith(".resume.json"):
            continue
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
            if path.is_file() and not path.name.endswith(".resume.json"):
                archive.write(path, path.relative_to(args.workdir))


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        cli.error("--max-hours must be finite and positive")
    if not math.isfinite(args.collect_reserve_minutes) or args.collect_reserve_minutes <= 0:
        cli.error("--collect-reserve-minutes must be finite and positive")
    if args.limit is not None and args.limit < 1:
        cli.error("--limit must be positive")
    if any(not math.isfinite(lam) or lam < 0 for lam in args.lam):
        cli.error("--lam values must be finite and nonnegative")
    if args.score_chunk_size < 1:
        cli.error("--score-chunk-size must be positive")
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
        if not problems_path.exists() or (args.refresh_problems and "collect" in requested):
            loader = load_math500 if args.dataset == "math500" else load_gsm8k
            save_jsonl(loader()[: args.limit], problems_path)
        for stage in STAGES:
            if stage not in requested:
                continue
            started = time.monotonic()
            remaining = deadline - started
            try:
                problem_ids(args)
                complete = not args.force and stage_complete(stage, args)
            except (OSError, ValueError, TypeError) as exc:
                status = f"failed: invalid problems.jsonl: {exc}"
                LOGGER.error(status)
                records.append((stage, status, time.monotonic() - started))
                failed = True
                continue
            if complete:
                status = "skipped (output exists)"
            elif remaining <= 0:
                status = "skipped (budget exhausted)"
                failed = True
            else:
                LOGGER.info("Starting %s with %.1f seconds remaining", stage, remaining)
                try:
                    input_count = trace_count(args)
                    fingerprint = merged_fingerprint(args.workdir)
                    if stage != "collect":
                        for output in stage_outputs(stage, args):
                            provenance_path(output).unlink(missing_ok=True)
                    backup = prepare_collection(args) if stage == "collect" else None
                    restoration = ""
                    try:
                        run_stage(stage, args, remaining)
                    finally:
                        if stage == "collect":
                            restoration = restore_collection(args, backup)
                            if restoration:
                                records.append(("collect recovery", restoration, 0.0))
                    if not all(path.exists() for path in stage_outputs(stage, args)):
                        raise ValueError("stage returned without expected outputs")
                    if stage != "collect":
                        for output in stage_outputs(stage, args):
                            provenance_path(output).write_text(
                                json.dumps(
                                    {"trace_count": input_count, "fingerprint": fingerprint}
                                ),
                                encoding="utf-8",
                            )
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
