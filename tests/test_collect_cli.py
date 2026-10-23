import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("explicit", [False, True])
def test_collect_flags(monkeypatch, tmp_path, explicit):
    spec = importlib.util.spec_from_file_location(
        "collect_cli", Path(__file__).resolve().parents[1] / "scripts" / "collect.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = {}

    def backend(model, **kwargs):
        calls["backend"] = (model, kwargs)
        return "stub"

    def collect(*args, **kwargs):
        calls["collect"] = (args, kwargs)

    monkeypatch.setattr(cli, "HFBackend", backend)
    monkeypatch.setattr(cli, "collect_dataset", collect)
    monkeypatch.setattr(cli, "load_math500", lambda path: ["problem"])
    monkeypatch.setattr(cli.time, "monotonic", lambda: 100)
    argv = ["collect", "--model", "tiny", "--dataset", "math500", "--out", str(tmp_path)]
    if explicit:
        argv += ["--dtype", "float16", "--attn-implementation", "eager", "--max-hours", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert calls["backend"] == (
        "tiny",
        dict(
            device="auto",
            seed=0,
            dtype="float16" if explicit else "auto",
            attn_implementation="eager" if explicit else "sdpa",
        ),
    )
    assert calls["collect"][0][0] == "stub"
    assert calls["collect"][1] == dict(resume=True, deadline=7300 if explicit else None)


@pytest.mark.parametrize("hours", ["0", "-1", "nan", "inf"])
def test_invalid_hours(monkeypatch, hours):
    import runpy

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect",
            "--model",
            "tiny",
            "--dataset",
            "math500",
            "--out",
            "unused",
            "--max-hours",
            hours,
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "scripts" / "collect.py"), run_name="__main__"
        )
