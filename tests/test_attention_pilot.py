import builtins
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from test_flips import make_trace

from osc.data.io import save_traceset
from osc.data.schema import TraceSet
from osc.pilots.attention_pilot import (
    AttentionError,
    attention_trajectory,
    compare_flip_vs_control,
    event_query_positions,
    run_attention_pilot,
)
from osc.pilots.flips import find_anchor_events


class Backend:
    def attention_to_span(self, prompt, thinking, span, query_char_positions):
        return np.tile(np.array([[[0.1, 0.3], [0.5, 0.7]]]), (len(query_char_positions), 1, 1))


def test_positions_reductions_and_runner():
    traces = TraceSet((make_trace(), make_trace((0, 1, 1, 1))))
    events = find_anchor_events(traces)
    assert event_query_positions(traces[0], events[0], 20) == [22]
    assert event_query_positions(traces[1], events[1], 1) == [22]
    args = (Backend(), "p", "abcd", (0, 1), [2, 3])
    np.testing.assert_allclose(attention_trajectory(*args), [0.4, 0.4])
    np.testing.assert_allclose(attention_trajectory(*args, layers=[-1]), [0.6, 0.6])
    np.testing.assert_allclose(attention_trajectory(*args, head_reduce="max"), [0.5, 0.5])
    assert attention_trajectory(Backend(), "", "", (0, 0), []).size == 0
    result = run_attention_pilot(Backend(), traces, {"p": "prompt"}, max_after=3)
    assert (result.n_flip, result.n_control, result.auroc) == (1, 1, 0.5)
    assert np.isnan(result.flip_curves[0, 1:]).all()
    with pytest.raises(AttentionError):
        run_attention_pilot(Backend(), traces, {})
    with pytest.raises(AttentionError):
        run_attention_pilot(Backend(), traces, {}, max_after=0)
    with pytest.raises(AttentionError):
        event_query_positions(traces[0], events[0], 0)


def test_matched_auc_and_permutation():
    flips = np.full((12, 3), 0.1)
    controls = np.full((12, 3), 0.9)
    flips[:6, 2] = np.nan
    result = compare_flip_vs_control(flips, controls, [2] * 6 + [3] * 6, n_perm=199)
    assert result.auroc == 1
    assert result.perm_p <= 0.01
    tied = compare_flip_vs_control(np.ones((3, 2)), np.ones((3, 2)), [1, 2, 2], n_perm=10)
    assert tied.auroc == 0.5
    assert tied.perm_p == 1
    missing = compare_flip_vs_control(np.full((1, 2), np.nan), controls[:, :2], [0])
    assert np.isnan(missing.auroc)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"head_reduce": "sum"},
        {"layers": []},
        {"layers": [2]},
        {"span": (-1, 2)},
        {"query_char_positions": [10]},
    ],
)
def test_invalid_trajectory(kwargs):
    args = dict(backend=Backend(), prompt="", thinking="abc", span=(0, 1), query_char_positions=[2])
    args.update(kwargs)
    with pytest.raises(AttentionError):
        attention_trajectory(**args)


def test_invalid_backend_and_curves():
    class Bad:
        def attention_to_span(self, *args):
            return np.zeros((2, 2))

    with pytest.raises(AttentionError):
        attention_trajectory(Bad(), "", "ab", (0, 1), [2])
    for flips, controls, offsets, n in [
        ([1], [[1]], [1], 2),
        ([[1]], [[1]], [], 2),
        ([[1]], [[1]], [0.5], 2),
        ([[1]], [[1]], [1], 0),
        ([[np.inf]], [[1]], [1], 2),
    ]:
        with pytest.raises(AttentionError):
            compare_flip_vs_control(flips, controls, offsets, n_perm=n)


@pytest.mark.parametrize("script", ["pilot_flips.py", "pilot_elicit.py"])
def test_cli_without_torch(script, tmp_path, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers"}:
            raise AssertionError("pilot must not import model dependencies")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    path = Path(__file__).resolve().parents[1] / "scripts" / script
    namespace = runpy.run_path(str(path))
    assert callable(namespace["main"])
    traces = tmp_path / "traces"
    save_traceset(TraceSet((make_trace(),)), traces)
    problems = tmp_path / "problems.jsonl"
    problems.write_text(json.dumps({"problem_id": "p", "prompt": "problem"}) + "\n")
    output = tmp_path / "result.json"

    class FakeHF(Backend):
        def __init__(self, model_name):
            assert model_name == "fake-model"

        def complete(self, *args):
            return "2"

    module = ModuleType("osc.collect.hf_backend")
    module.HFBackend = FakeHF
    monkeypatch.setitem(sys.modules, "osc.collect.hf_backend", module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            script,
            "--traces",
            str(traces),
            "--problems",
            str(problems),
            "--model",
            "fake-model",
            "--out",
            str(output),
        ],
    )
    namespace["main"]()
    result = json.loads(output.read_text())
    if script == "pilot_flips.py":
        assert result["n_flip"] == 1
        assert result["auroc"] is None
    else:
        assert result["n_traces"] == 1
        assert result["parse_rate"] == 1
