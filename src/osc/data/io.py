"""Save/load TraceSets without pickle: text fields in JSONL, arrays in a single .npz."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from osc.data.schema import SchemaError, SiblingSet, Trace, TraceSet

_META_FILE = "metadata.json"
_TEXT_FILE = "traces.jsonl"
_ARRAY_FILE = "arrays.npz"
_FORMAT_VERSION = 1


def save_traceset(traces: TraceSet, directory: str | Path) -> Path:
    """Write ``traces`` into ``directory`` (created if needed). Returns the directory."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    with (out / _TEXT_FILE).open("w", encoding="utf-8") as fh:
        for i, tr in enumerate(traces):
            arrays[f"{i}/step_tokens"] = tr.step_tokens
            arrays[f"{i}/forced_correct"] = tr.forced_correct
            arrays[f"{i}/features"] = tr.features
            sibling_steps = []
            for t, sib in enumerate(tr.siblings):
                if sib is None:
                    continue
                sibling_steps.append(t)
                arrays[f"{i}/sib/{t}/step_tokens"] = sib.step_tokens
                arrays[f"{i}/sib/{t}/forced_correct"] = sib.forced_correct
                arrays[f"{i}/sib/{t}/features"] = sib.features
            record = {
                "index": i,
                "problem_id": tr.problem_id,
                "step_texts": list(tr.step_texts),
                "forced_answers": list(tr.forced_answers),
                "natural_stop": tr.natural_stop,
                "final_correct": tr.final_correct,
                "has_sibling_slots": bool(tr.siblings),
                "sibling_steps": sibling_steps,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    np.savez_compressed(out / _ARRAY_FILE, **arrays)
    meta = {"format_version": _FORMAT_VERSION, "metadata": dict(traces.metadata)}
    (out / _META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def load_traceset(directory: str | Path) -> TraceSet:
    """Load a TraceSet written by :func:`save_traceset`."""
    src = Path(directory)
    for name in (_META_FILE, _TEXT_FILE, _ARRAY_FILE):
        if not (src / name).is_file():
            raise FileNotFoundError(f"missing {name} in {src}")
    meta = json.loads((src / _META_FILE).read_text(encoding="utf-8"))
    if meta.get("format_version") != _FORMAT_VERSION:
        raise SchemaError(f"unsupported trace format version {meta.get('format_version')!r}")
    traces = []
    with np.load(src / _ARRAY_FILE, allow_pickle=False) as arrays:
        with (src / _TEXT_FILE).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    traces.append(_trace_from_record(json.loads(line), arrays))
    return TraceSet(tuple(traces), meta.get("metadata", {}))


def _trace_from_record(record: dict[str, Any], arrays: Any) -> Trace:
    i = record["index"]
    n_steps = len(record["step_texts"])
    siblings: tuple[SiblingSet | None, ...] = ()
    if record.get("has_sibling_slots"):
        slots: list[SiblingSet | None] = [None] * n_steps
        for t in record.get("sibling_steps", []):
            slots[t] = SiblingSet(
                step_tokens=arrays[f"{i}/sib/{t}/step_tokens"],
                forced_correct=arrays[f"{i}/sib/{t}/forced_correct"],
                features=arrays[f"{i}/sib/{t}/features"],
            )
        siblings = tuple(slots)
    return Trace(
        problem_id=record["problem_id"],
        step_texts=tuple(record["step_texts"]),
        step_tokens=arrays[f"{i}/step_tokens"],
        forced_answers=tuple(record["forced_answers"]),
        forced_correct=arrays[f"{i}/forced_correct"],
        features=arrays[f"{i}/features"],
        natural_stop=record["natural_stop"],
        final_correct=record["final_correct"],
        siblings=siblings,
    )
