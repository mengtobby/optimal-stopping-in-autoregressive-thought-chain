"""Causal, delimiter-preserving reasoning segmentation."""

import re
from typing import Literal


class SegmentationError(ValueError):
    """Invalid segmentation options."""


_MARKER = re.compile(
    r"\s*(?:Wait|Alternatively|Hmm|But wait|Let me check|Let me verify|Actually)\b"
)


def segment_thinking(
    thinking: str,
    mode: Literal["paragraph", "marker"] = "paragraph",
    min_chars: int = 40,
    max_steps: int | None = 64,
) -> list[tuple[int, int]]:
    if mode not in ("paragraph", "marker"):
        raise SegmentationError("mode must be paragraph or marker")
    if min_chars < 0 or (max_steps is not None and max_steps < 1):
        raise SegmentationError("min_chars must be nonnegative and max_steps positive")
    spans = []
    start = 0
    for match in re.finditer(r"\n\n", thinking):
        end = match.end()
        if end == len(thinking):
            continue
        if max_steps is not None and len(spans) >= max_steps - 1:
            break
        if end - start >= min_chars and (mode == "paragraph" or _MARKER.match(thinking[end:])):
            spans.append((start, end))
            start = end
    if start < len(thinking):
        spans.append((start, len(thinking)))
    return spans


def first_step(continuation: str, mode: str = "paragraph", min_chars: int = 40) -> str:
    spans = segment_thinking(continuation, mode, min_chars, None)
    return continuation[: spans[0][1]] if spans else continuation
