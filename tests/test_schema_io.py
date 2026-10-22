import numpy as np
import pytest

from osc.data.io import load_traceset, save_traceset
from osc.data.schema import SchemaError, SiblingSet, Trace, TraceSet


def _sibs(k: int = 3, d: int = 2) -> SiblingSet:
    return SiblingSet(
        step_tokens=np.arange(1, k + 1),
        forced_correct=np.linspace(0, 1, k),
        features=np.ones((k, d)),
    )


def _trace(pid: str = "p0", t: int = 3, d: int = 2, siblings: bool = True) -> Trace:
    return Trace(
        problem_id=pid,
        step_texts=tuple(f"step {i}" for i in range(t)),
        step_tokens=np.array([0] + [10] * (t - 1)),
        forced_answers=tuple(["1"] * t),
        forced_correct=np.linspace(0.2, 0.9, t),
        features=np.arange(t * d, dtype=float).reshape(t, d),
        natural_stop=True,
        final_correct=1.0,
        siblings=tuple([_sibs(d=d)] * (t - 1) + [None]) if siblings else (),
    )


def test_trace_arrays_are_read_only_copies():
    source = np.array([0, 5, 5])
    tr = Trace("p", ("a", "b", "c"), source, (None,) * 3, np.zeros(3), np.zeros((3, 1)), False, 0.0)
    source[0] = 99
    assert tr.step_tokens[0] == 0
    with pytest.raises(ValueError):
        tr.step_tokens[0] = 1
    assert list(tr.cum_tokens) == [0, 5, 10]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forced_correct": np.array([0.0, 1.5, 0.0])},
        {"step_tokens": np.array([0, -1, 2])},
        {"features": np.zeros((2, 2))},
        {"step_texts": ("a",)},
        {"final_correct": 2.0},
    ],
)
def test_trace_validation(kwargs):
    base = {
        "problem_id": "p",
        "step_texts": ("a", "b", "c"),
        "step_tokens": np.array([0, 1, 2]),
        "forced_answers": (None, None, None),
        "forced_correct": np.zeros(3),
        "features": np.zeros((3, 2)),
        "natural_stop": False,
        "final_correct": 0.0,
    }
    with pytest.raises(SchemaError):
        Trace(**{**base, **kwargs})


def test_sibling_validation():
    with pytest.raises(SchemaError):
        SiblingSet(np.array([1]), np.array([0.5]), np.zeros((1, 2)))
    with pytest.raises(SchemaError):
        _trace(siblings=False).__class__(
            **{**_trace(siblings=False).__dict__, "siblings": (_sibs(),) * 3}
        )


def test_traceset_requires_consistent_dims_and_is_immutable():
    with pytest.raises(SchemaError):
        TraceSet((_trace(d=2), _trace(d=3)))
    with pytest.raises(SchemaError):
        TraceSet(())
    ts = TraceSet((_trace("a"), _trace("b", t=5)), {"model": "m"})
    assert ts.max_steps == 5 and len(ts) == 2
    with pytest.raises(TypeError):
        ts.metadata["model"] = "x"  # type: ignore[index]
    assert ts.subset([1])[0].problem_id == "b"
    assert ts.with_metadata(seed=1).metadata == {"model": "m", "seed": 1}


def test_roundtrip(tmp_path):
    ts = TraceSet((_trace("a"), _trace("b", t=4, siblings=False)), {"lambda": 0.1})
    loaded = load_traceset(save_traceset(ts, tmp_path / "out"))
    assert loaded.metadata == {"lambda": 0.1}
    for a, b in zip(ts, loaded, strict=True):
        assert a.problem_id == b.problem_id and a.step_texts == b.step_texts
        np.testing.assert_array_equal(a.features, b.features)
        np.testing.assert_array_equal(a.forced_correct, b.forced_correct)
        assert len(a.siblings) == len(b.siblings)
        for sa, sb in zip(a.siblings, b.siblings, strict=True):
            assert (sa is None) == (sb is None)
            if sa is not None:
                np.testing.assert_array_equal(sa.features, sb.features)


def test_load_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_traceset(tmp_path)
