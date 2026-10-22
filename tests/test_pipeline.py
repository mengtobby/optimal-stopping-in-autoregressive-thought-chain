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
    assert all(s is None for s in trace.siblings[-2:])
    assert backend.hidden_features_calls == 1 + 2 * (trace.n_steps - 2)
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
