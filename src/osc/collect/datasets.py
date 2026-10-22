"""Local JSONL problems and optional Hugging Face dataset adapters."""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


class DatasetError(ValueError):
    """A problem record is malformed."""


@dataclass(frozen=True)
class Problem:
    problem_id: str
    prompt: str
    answer: str
    source: str

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise DatasetError(f"{key} must be a nonempty string")


def load_jsonl(path: str | Path) -> tuple[Problem, ...]:
    problems = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                problems.append(Problem(**json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise DatasetError(f"{path}:{number}: invalid problem: {exc}") from exc
    return tuple(problems)


def save_jsonl(problems: Iterable[Problem], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for problem in problems:
            handle.write(json.dumps(asdict(problem), ensure_ascii=False) + "\n")


def load_math500(path: str | Path | None = None) -> tuple[Problem, ...]:
    if path is not None:
        return load_jsonl(path)
    from datasets import load_dataset

    rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return tuple(Problem(str(r["unique_id"]), r["problem"], r["answer"], "math500") for r in rows)


def load_gsm8k(path: str | Path | None = None, split: str = "test") -> tuple[Problem, ...]:
    if path is not None:
        return load_jsonl(path)
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split=split)
    problems = []
    for i, row in enumerate(rows):
        if "####" not in row["answer"]:
            raise DatasetError(f"gsm8k row {i} has no final answer delimiter")
        problems.append(
            Problem(
                f"{split}-{i}", row["question"], row["answer"].rsplit("####", 1)[1].strip(), "gsm8k"
            )
        )
    return tuple(problems)
