import pytest

from osc.collect.datasets import Problem
from osc.collect.fake_backend import FakeBackend
from osc.collect.pipeline import CollectConfig, CollectionError, collect_dataset, collect_trace
from osc.data.io import load_traceset


def test_trace():
    backend = FakeBackend()
    config = CollectConfig(min_chars=0, k_siblings=2)
    trace = collect_trace(backend, Problem("a", "q", "1", "test"), config)
    assert "".join(trace.step_texts) == backend.thinking
    assert trace.step_texts[0] == "" and trace.step_tokens[0] == 0
    assert trace.forced_correct.tolist() == [1] * trace.n_steps
    assert trace.siblings[-2] is not None and trace.siblings[-1] is None
    assert backend.hidden_features_calls == 1 + 2 * (trace.n_steps - 1)
    backend = FakeBackend()
    collect_trace(backend, Problem("a", "q", "1", "test"), CollectConfig(k_siblings=0))
    assert backend.hidden_features_calls == 1
    assert backend.complete("q", "", "", 1) == "1"
    assert backend.attention_to_span("q", "abc", (0, 2), [0, 3]).shape == (2, 1, 1)


def test_resume_failure(tmp_path):
    good = Problem("a", "q", "1", "test")
    bad = Problem("b", "bad", "1", "test")
    backend = FakeBackend(thinking={"q": "a\n\nb\n\nc"})
    config = CollectConfig(min_chars=0, k_siblings=0)
    assert len(collect_dataset(backend, [good, bad], config, tmp_path)) == 1
    backend.thinking = {}
    assert len(collect_dataset(backend, [good], config, tmp_path)) == 1
    assert load_traceset(tmp_path / "merged")[0].problem_id == "a"
    with pytest.raises(CollectionError):
        collect_dataset(backend, [bad], config, tmp_path)


@pytest.mark.parametrize("kwargs", [{"n_forced": 0}, {"k_siblings": 1}, {"min_chars": -1}])
def test_config(kwargs):
    with pytest.raises(ValueError):
        CollectConfig(**kwargs)


def test_merged_final_has_no_siblings():
    trace = collect_trace(
        FakeBackend(),
        Problem("a", "q", "1", "test"),
        CollectConfig(min_chars=0, max_steps=2, k_siblings=2),
    )
    assert trace.siblings[0] is not None
    assert trace.siblings[-2] is None


def test_extension_and_ended_thinking():
    from osc.collect.pipeline import collect_trace_with_stats

    class Scripted(FakeBackend):
        def continue_steps(self, prompt, thinking_prefix, k, max_tokens):
            if k == 2:
                assert max_tokens == 4
                return ["abcd", "x"]
            assert thinking_prefix == "abcd"
            return ["\n\nzz"]

    trace, stats = collect_trace_with_stats(
        Scripted(thinking="last"),
        Problem("a", "q", "1", "test"),
        CollectConfig(min_chars=0, k_siblings=2, sibling_max_tokens=4),
    )
    assert trace.siblings[0].step_tokens.tolist() == [6, 1]
    assert stats.n_extended == 1
    assert stats.n_ended_thinking == 1
    assert stats.n_draws == 2


def test_budget_and_stride():
    from osc.collect.pipeline import collect_trace_with_stats

    trace, stats = collect_trace_with_stats(
        FakeBackend(thinking="a\n\nb", continuations=("xxxxxxxxxx",)),
        Problem("a", "q", "1", "test"),
        CollectConfig(
            min_chars=0, max_thinking_tokens=5, sibling_max_tokens=3, k_siblings=2, sibling_stride=1
        ),
    )
    assert trace.siblings[0].step_tokens.tolist() == [5, 5]
    assert trace.siblings[1].step_tokens.tolist() == [2, 2]
    assert stats.n_budget_truncated == 4


def test_corrupt_shard_and_atomic_write(tmp_path, monkeypatch, caplog):
    import osc.collect.pipeline as pipeline

    problem = Problem("a", "q", "1", "test")
    config = CollectConfig(k_siblings=0)
    replacements = []
    original = pipeline.os.replace

    def replace(source, destination):
        assert source.parent == destination.parent
        assert (source / "metadata.json").exists()
        replacements.append(destination)
        original(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", replace)
    first = collect_dataset(FakeBackend(), [problem], config, tmp_path)
    (replacements[0] / "arrays.npz").write_bytes(b"corrupt")
    resumed = collect_dataset(FakeBackend(), [problem], config, tmp_path)
    assert len(replacements) == 2
    assert "invalid shard" in caplog.text
    assert resumed.metadata["sibling_stats"] == first.metadata["sibling_stats"]


def test_pipeline_import_is_lazy():
    import os
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import osc.collect.pipeline; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
    )
