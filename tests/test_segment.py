import random

import pytest

from osc.collect.segment import SegmentationError, first_step, segment_thinking


@pytest.mark.parametrize("mode", ["paragraph", "marker"])
@pytest.mark.parametrize("minimum", [0, 4, 40])
@pytest.mark.parametrize("seed", range(24))
def test_causal_spans(mode, minimum, seed):
    text = "Start here.\n\nWait, more.\n\nplain.\n\nActually yes.\n\nEnd."
    rng = random.Random(seed)
    text += "\n\n" + "\n\n".join(
        rng.choice(
            [
                "Wait, yes.",
                "Actually no.",
                "Hmm",
                "x",
                "plain paragraph " * 5,
                "Alternatively try again.",
                "Let me check.",
            ]
        )
        for _ in range(rng.randint(2, 20))
    )
    spans = segment_thinking(text, mode, minimum, None)
    assert "".join(text[a:b] for a, b in spans) == text
    assert all(b == c for (_, b), (c, _) in zip(spans, spans[1:], strict=False))
    for (_, end), (a, b) in zip(spans, spans[1:], strict=False):
        assert first_step(text[end:], mode, minimum) == text[a:b]
    capped = segment_thinking(text, mode, minimum, 2)
    assert len(capped) <= 2
    assert "".join(text[a:b] for a, b in capped) == text


def test_edges():
    assert segment_thinking("") == []
    assert first_step("abc") == "abc"
    assert segment_thinking("a\n\nb", min_chars=0) == [(0, 3), (3, 4)]
    assert segment_thinking("a\n\n", min_chars=0) == [(0, 3)]
    for kwargs in ({"mode": "bad"}, {"min_chars": -1}, {"max_steps": 0}):
        with pytest.raises(SegmentationError):
            segment_thinking("abc", **kwargs)
