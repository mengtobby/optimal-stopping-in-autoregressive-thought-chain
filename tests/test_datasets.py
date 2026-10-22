import sys
import types

import pytest

from osc.collect.datasets import (
    DatasetError,
    Problem,
    load_gsm8k,
    load_jsonl,
    load_math500,
    save_jsonl,
)


def test_local(tmp_path):
    path = tmp_path / "data.jsonl"
    problems = (Problem("a", "question", "1", "test"),)
    save_jsonl(problems, path)
    assert load_jsonl(path) == problems
    assert load_math500(path) == problems
    assert load_gsm8k(path) == problems
    path.write_text('\n{"bad": 1}\n')
    with pytest.raises(DatasetError, match=":2:"):
        load_jsonl(path)
    with pytest.raises(DatasetError):
        Problem("", "x", "1", "test")


def test_remote(monkeypatch):
    def load(name, *args, **kwargs):
        if "MATH" in name:
            return [{"unique_id": "a", "problem": "q", "answer": "1"}]
        return [{"question": "q", "answer": "work #### 42"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load))
    assert load_math500()[0].problem_id == "a"
    assert load_gsm8k()[0].answer == "42"
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda *a, **k: [{"question": "q", "answer": "bad"}]),
    )
    with pytest.raises(DatasetError):
        load_gsm8k()
