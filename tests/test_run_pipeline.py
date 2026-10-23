import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from osc.collect.datasets import Problem
from osc.data.io import load_traceset, save_traceset
from osc.data.schema import Trace, TraceSet

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cloud_pipeline", ROOT / "scripts/run_pipeline.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
RUN_STAGE = runner.run_stage


def traces_for(*ids: str) -> TraceSet:
    return TraceSet(tuple(Trace(pid, ("",), [0], ("1",), [1], [[0]], True, 1) for pid in ids))


@pytest.fixture
def cloud(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_math500", lambda: (Problem("p", "question", "1", "test"),))
    calls = []

    def stage(name, args, remaining):
        calls.append((name, remaining))
        if name == "collect":
            ids = []
            for line in (args.workdir / "problems.jsonl").read_text().splitlines():
                problem_id = json.loads(line)["problem_id"]
                ids.append(problem_id)
                shard = (
                    args.workdir / "traces/shards" / hashlib.sha256(problem_id.encode()).hexdigest()
                )
                save_traceset(traces_for(problem_id), shard)
            save_traceset(traces_for(*ids), args.workdir / "traces/merged")
            return
        for path in runner.stage_outputs(name, args):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"headroom": 0.25, "n_flip": 2}), encoding="utf-8")

    monkeypatch.setattr(runner, "run_stage", stage)
    return tmp_path, calls, stage


def test_order_resume_force_and_archive(cloud):
    work, calls, _ = cloud
    flags = ["--workdir", str(work), "--stages", "flips,headroom,collect,elicit"]
    assert runner.main(flags) == 0
    assert [name for name, _ in calls] == list(runner.STAGES)
    assert (work / "problems.jsonl").exists()
    summary = (work / "SUMMARY.md").read_text()
    assert "headroom: 0.25" in summary
    assert "n_flip: 2" in summary
    with zipfile.ZipFile(work / "results.zip") as archive:
        assert "SUMMARY.md" in archive.namelist()
        assert "results/pilot_elicit.json" in archive.namelist()
        assert not any("traces/" in name for name in archive.namelist())
    calls.clear()
    assert runner.main(flags) == 0
    assert calls == []
    assert "skipped (output exists)" in (work / "SUMMARY.md").read_text()
    assert runner.main([*flags, "--force"]) == 0
    assert len(calls) == 4


def test_failure_isolation(cloud, monkeypatch):
    work, calls, stage = cloud

    def failing(name, args, remaining):
        if name == "headroom":
            raise ValueError("bad traces")
        stage(name, args, remaining)

    monkeypatch.setattr(runner, "run_stage", failing)
    assert runner.main(["--workdir", str(work)]) == 1
    assert [name for name, _ in calls] == ["collect", "elicit", "flips"]
    assert "failed: ValueError: bad traces" in (work / "SUMMARY.md").read_text()
    assert (work / "results.zip").exists()


def test_budget_propagates_and_exhausts(cloud, monkeypatch):
    work, calls, stage = cloud
    now = [100.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: now[0])

    def advancing(name, args, remaining):
        stage(name, args, remaining)
        now[0] += 2

    monkeypatch.setattr(runner, "run_stage", advancing)
    assert runner.main(["--workdir", str(work), "--max-hours", str(3 / 3600)]) == 1
    assert calls == [("collect", 3), ("headroom", 1)]
    assert "budget exhausted" in (work / "SUMMARY.md").read_text()


def test_subprocess_timeout_and_collection_options(cloud, monkeypatch):
    work, _, _ = cloud
    commands = []
    monkeypatch.setattr(
        runner.subprocess, "run", lambda command, **kw: commands.append((command, kw))
    )
    args = runner.parser().parse_args(
        ["--workdir", str(work), "--force", "--score-chunk-size", "3"]
    )
    for name in runner.STAGES:
        RUN_STAGE(name, args, 60)
    command, options = commands[0]
    assert command[0] == sys.executable
    assert command[command.index("--attn-implementation") + 1] == "sdpa"
    assert command[command.index("--dtype") + 1] == "auto"
    assert float(command[command.index("--max-hours") + 1]) * 3600 < 60
    assert "--no-resume" in command
    assert all(options == {"check": True, "timeout": 60} for _, options in commands)
    assert "--pilot-worker" in commands[2][0]
    assert commands[2][0][-2:] == ["--score-chunk-size", "3"]


def test_timeout_is_reported(cloud, monkeypatch):
    work, _, _ = cloud

    def timeout(*args):
        raise subprocess.TimeoutExpired("stage", 1)

    monkeypatch.setattr(runner, "run_stage", timeout)
    assert runner.main(["--workdir", str(work)]) == 1
    assert "TimeoutExpired" in (work / "SUMMARY.md").read_text()


@pytest.mark.parametrize("remaining,reserve_minutes", [(7200, 30), (39600, 30), (39600, 180)])
def test_collection_soft_deadline_reserves_wind_down(monkeypatch, remaining, reserve_minutes):
    commands = []
    monkeypatch.setattr(
        runner.subprocess, "run", lambda command, **kw: commands.append((command, kw))
    )
    args = runner.parser().parse_args(
        ["--workdir", ".", "--collect-reserve-minutes", str(reserve_minutes)]
    )
    RUN_STAGE("collect", args, remaining)
    command, options = commands[0]
    soft_seconds = float(command[command.index("--max-hours") + 1]) * 3600
    assert remaining - soft_seconds >= reserve_minutes * 60
    assert remaining - soft_seconds >= remaining * 0.25
    assert 0 < soft_seconds < remaining
    assert options["timeout"] == remaining


@pytest.mark.parametrize("remaining", [60, 0.5])
def test_collection_short_budget_has_positive_soft_window(monkeypatch, remaining):
    commands = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kw: commands.append(command))
    args = runner.parser().parse_args(["--workdir", "."])
    RUN_STAGE("collect", args, remaining)
    soft_seconds = float(commands[0][commands[0].index("--max-hours") + 1]) * 3600
    assert soft_seconds == pytest.approx(min(1.0, remaining * 0.1))
    assert 0 < soft_seconds < remaining


def test_collection_can_finish_in_flight_problem_and_run_pilots(cloud, monkeypatch):
    work, _, _ = cloud
    now = [0.0]
    calls = []
    monkeypatch.setattr(runner.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(runner, "run_stage", RUN_STAGE)
    args = runner.parser().parse_args(["--workdir", str(work)])

    def subprocess_run(command, *, check, timeout):
        stage = runner.STAGES[len(calls)]
        calls.append(stage)
        if stage == "collect":
            soft_seconds = float(command[command.index("--max-hours") + 1]) * 3600
            # Finish a ten-minute problem started just before the soft deadline,
            # then merge shards. This would exceed the former five-second margin.
            elapsed = soft_seconds + 600 + 10
        else:
            elapsed = 60
        assert check
        assert elapsed < timeout
        now[0] += elapsed
        for path in runner.stage_outputs(stage, args):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(runner.subprocess, "run", subprocess_run)
    assert runner.main(["--workdir", str(work), "--max-hours", "11"]) == 0
    assert calls == list(runner.STAGES)
    assert "failed" not in (work / "SUMMARY.md").read_text()


@pytest.mark.parametrize(
    "flags",
    [
        ["--max-hours", "nan"],
        ["--max-hours", "0"],
        ["--collect-reserve-minutes", "nan"],
        ["--collect-reserve-minutes", "inf"],
        ["--collect-reserve-minutes", "0"],
        ["--collect-reserve-minutes", "-1"],
        ["--limit", "0"],
        ["--keep-backups", "-1"],
        ["--stages", "bad"],
        ["--stages", "collect,collect"],
        ["--lam", "-1"],
        ["--n-forced", "0"],
    ],
)
def test_validation(flags):
    with pytest.raises(SystemExit):
        runner.main(flags)


@pytest.mark.parametrize("name", ["kaggle_run", "colab_run"])
def test_notebooks(name):
    notebook = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 0
    assert isinstance(notebook["metadata"], dict)
    cells = notebook["cells"]
    assert any(cell["cell_type"] == "markdown" for cell in cells)
    assert any(
        cell["cell_type"] == "code" and "scripts/run_pipeline.py" in "".join(cell["source"])
        for cell in cells
    )
    for cell in cells:
        assert isinstance(cell["source"], list)
        assert isinstance(cell["metadata"], dict)
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


@pytest.mark.parametrize("stage,attention", [("elicit", "sdpa"), ("flips", "eager")])
def test_pilot_worker_backend_options(monkeypatch, stage, attention):
    from osc.collect import hf_backend

    captured = []
    monkeypatch.setattr(hf_backend, "HFBackend", lambda *a, **kw: captured.append((a, kw)))
    monkeypatch.setattr(sys, "argv", [])

    def script(path, run_name):
        assert Path(path).name == f"pilot_{stage}.py"
        assert run_name == "__main__"
        hf_backend.HFBackend("model", "cpu", score_chunk_size=3, temperature=0.2)

    monkeypatch.setattr(runner.runpy, "run_path", script)
    runner.pilot_worker([stage, "float16", "7", "--model", "model"])
    assert captured == [
        (
            ("model", "cpu"),
            {
                "dtype": "float16",
                "seed": 7,
                "attn_implementation": attention,
                "score_chunk_size": 3,
                "temperature": 0.2,
            },
        )
    ]


def test_partial_shards_recollect_and_reports_follow_growth(cloud):
    work, calls, _ = cloud
    flags = ["--workdir", str(work)]
    assert runner.main(flags) == 0
    calls.clear()
    # Add a pending problem while keeping the previous merged subset and reports.
    with (work / "problems.jsonl").open("a") as stream:
        stream.write(
            json.dumps({"problem_id": "pending", "prompt": "q", "answer": "1", "source": "test"})
            + "\n"
        )
    assert runner.main([*flags, "--stages", "collect"]) == 0
    assert [name for name, _ in calls] == ["collect"]
    assert "Collected 2 of 2 problems" in (work / "SUMMARY.md").read_text()
    calls.clear()
    assert runner.main(flags) == 0
    assert [name for name, _ in calls] == ["headroom", "elicit", "flips"]
    calls.clear()
    assert runner.main(flags) == 0
    assert calls == []


@pytest.mark.parametrize(
    "stages,refresh,changed",
    [("collect", False, False), ("elicit", True, False), ("collect", True, True)],
)
def test_force_preserves_problems_unless_explicitly_refreshed(
    cloud, monkeypatch, stages, refresh, changed
):
    work, _, _ = cloud
    assert runner.main(["--workdir", str(work)]) == 0
    path = work / "problems.jsonl"
    original = path.read_bytes()
    monkeypatch.setattr(runner, "load_math500", lambda: (Problem("new", "other", "2", "test"),))
    flags = ["--workdir", str(work), "--stages", stages, "--force"]
    if refresh:
        flags.append("--refresh-problems")
    assert runner.main(flags) == 0
    assert (path.read_bytes() != original) == changed


@pytest.mark.parametrize(
    "damage",
    ["subset", "old_arrays", "truncated_arrays", "wrong_ids", "duplicate_ids", "invalid_json"],
)
def test_all_shards_with_interrupted_merge_are_remerged(cloud, monkeypatch, damage):
    work, calls, _ = cloud
    monkeypatch.setattr(
        runner,
        "load_math500",
        lambda: (Problem("p", "question", "1", "test"), Problem("q", "other", "2", "test")),
    )
    flags = ["--workdir", str(work)]
    assert runner.main(flags) == 0
    merged = work / "traces/merged"
    full_json = (merged / "traces.jsonl").read_bytes()
    if damage in ("subset", "old_arrays"):
        save_traceset(traces_for("p"), merged)
        if damage == "old_arrays":
            (merged / "traces.jsonl").write_bytes(full_json)
    elif damage == "truncated_arrays":
        (merged / "arrays.npz").write_bytes(b"PK")
    elif damage == "wrong_ids":
        save_traceset(traces_for("p", "unexpected"), merged)
    elif damage == "duplicate_ids":
        save_traceset(traces_for("p", "p"), merged)
    else:
        (merged / "traces.jsonl").write_text('{"problem_id":')
    args = runner.parser().parse_args(flags)
    assert runner.collection_progress(args) == (2, 2)
    calls.clear()
    assert runner.main(flags) == 0
    assert [name for name, _ in calls] == ["collect"]
    assert {trace.problem_id for trace in load_traceset(merged)} == {"p", "q"}
    calls.clear()
    assert runner.main(flags) == 0
    assert calls == []


@pytest.mark.parametrize("change", ["metadata", "arrays", "text", "legacy"])
def test_content_changes_invalidate_reports(cloud, change):
    work, calls, _ = cloud
    flags = ["--workdir", str(work)]
    assert runner.main(flags) == 0
    merged = work / "traces/merged"
    if change == "metadata":
        path = merged / "metadata.json"
        payload = json.loads(path.read_text())
        payload["metadata"]["revision"] = 2
        path.write_text(json.dumps(payload))
    elif change == "legacy":
        for path in (work / "results").glob("*.resume.json"):
            path.write_text(json.dumps({"trace_count": 1}))
    else:
        trace = load_traceset(merged)[0]
        if change == "arrays":
            trace = replace(trace, features=trace.features + 7)
            save_traceset(TraceSet((trace,)), merged)
        else:
            path = merged / "traces.jsonl"
            payload = json.loads(path.read_text())
            payload["step_texts"] = ["changed"]
            path.write_text(json.dumps(payload) + "\n")
    calls.clear()
    assert runner.main(flags) == 0
    assert [name for name, _ in calls] == ["headroom", "elicit", "flips"]
    calls.clear()
    assert runner.main(flags) == 0
    assert calls == []
    assert "resume.json" not in (work / "SUMMARY.md").read_text()
    with zipfile.ZipFile(work / "results.zip") as archive:
        assert not any(name.endswith(".resume.json") for name in archive.namelist())


@pytest.mark.parametrize("force", [False, True])
def test_collection_never_shrinks_merged(cloud, monkeypatch, force):
    work, _, _ = cloud
    save_traceset(traces_for("p", "q"), work / "traces/merged")
    monkeypatch.setattr(
        runner,
        "load_math500",
        lambda: (Problem("p", "question", "1", "test"), Problem("q", "other", "2", "test")),
    )

    def smaller(name, args, remaining):
        save_traceset(traces_for("p"), args.workdir / "traces/merged")

    monkeypatch.setattr(runner, "run_stage", smaller)
    flags = ["--workdir", str(work), "--stages", "collect"]
    assert runner.main(flags + (["--force"] if force else [])) == 0
    assert len(load_traceset(work / "traces/merged")) == 2
    assert len(list((work / "traces").glob("merged.bak-*"))) == 1
    partial = next((work / "traces").glob("merged.partial-*"))
    assert len(load_traceset(partial)) == 1
    assert "restored 2 problems" in (work / "SUMMARY.md").read_text()


def test_force_moves_shards_before_collection(cloud, monkeypatch):
    work, _, _ = cloud
    old = work / "traces/shards/old"
    old.mkdir(parents=True)
    (old / "marker").write_text("saved")

    def interrupted(name, args, remaining):
        assert not (args.workdir / "traces/shards").exists()
        raise ValueError("early stop")

    monkeypatch.setattr(runner, "run_stage", interrupted)
    assert runner.main(["--workdir", str(work), "--stages", "collect", "--force"]) == 1
    saved = next((work / "traces").glob("shards.old-*"))
    assert (saved / "old/marker").read_text() == "saved"


@pytest.mark.parametrize("content", ['{"problem_id":', '{"problem_id": "p"}', "[]"])
def test_malformed_problems_write_failure_summary(cloud, content):
    work, calls, _ = cloud
    (work / "problems.jsonl").write_text(content + "\n")
    assert runner.main(["--workdir", str(work)]) == 1
    assert calls == []
    summary = (work / "SUMMARY.md").read_text()
    assert "failed: invalid problems.jsonl" in summary
    assert "Could not read problems.jsonl" in summary


@pytest.mark.parametrize("new_id", ["p", "replacement"])
def test_refresh_keeps_smaller_current_dataset_and_resume_settles(cloud, monkeypatch, new_id):
    work, calls, _ = cloud
    monkeypatch.setattr(
        runner,
        "load_math500",
        lambda: (Problem("p", "question", "1", "test"), Problem("q", "other", "2", "test")),
    )
    flags = ["--workdir", str(work)]
    assert runner.main(flags) == 0
    monkeypatch.setattr(
        runner, "load_math500", lambda: (Problem(new_id, "new question", "3", "test"),)
    )
    calls.clear()
    assert runner.main([*flags, "--stages", "collect", "--refresh-problems", "--force"]) == 0
    assert [name for name, _ in calls] == ["collect"]
    assert {trace.problem_id for trace in load_traceset(work / "traces/merged")} == {new_id}
    backup = next((work / "traces").glob("merged.bak-*"))
    assert len(load_traceset(backup)) == 2
    assert not list((work / "traces").glob("merged.partial-*"))
    assert "restored" not in (work / "SUMMARY.md").read_text()
    calls.clear()
    assert runner.main(flags) == 0
    assert [name for name, _ in calls] == ["headroom", "elicit", "flips"]
    calls.clear()
    assert runner.main(flags) == 0
    assert calls == []
    assert list((work / "traces").glob("merged.bak-*")) == [backup]


def test_ordinary_partial_resume_creates_no_backup(cloud):
    work, _, _ = cloud
    flags = ["--workdir", str(work)]
    assert runner.main(flags) == 0
    with (work / "problems.jsonl").open("a") as stream:
        stream.write(
            json.dumps({"problem_id": "q", "prompt": "q", "answer": "1", "source": "test"}) + "\n"
        )
    assert runner.main(flags) == 0
    assert not list((work / "traces").glob("merged.bak-*"))


@pytest.mark.parametrize("keep", [0, 1, 2])
def test_success_prunes_oldest_backups(cloud, caplog, keep):
    import os

    work, _, _ = cloud
    flags = ["--workdir", str(work), "--stages", "collect"]
    assert runner.main(flags) == 0
    for label in ("bak", "partial"):
        for index in range(3):
            path = work / "traces" / f"merged.{label}-{index}"
            path.mkdir()
            os.utime(path, ns=(index + 1, index + 1))
    # A pending problem makes collection run without needing a new backup.
    (work / "traces/merged/arrays.npz").unlink()
    with caplog.at_level("INFO"):
        assert runner.main([*flags, "--keep-backups", str(keep)]) == 0
    for label in ("bak", "partial"):
        assert sorted(path.name for path in (work / "traces").glob(f"merged.{label}-*")) == [
            f"merged.{label}-{index}" for index in range(3 - keep, 3)
        ]
    assert "Removed old merged" in caplog.text


def test_fingerprint_loads_once_per_state(cloud, monkeypatch):
    work, _, stage = cloud
    flags = ["--workdir", str(work), "--stages", "collect"]
    assert runner.main(flags) == 0
    loads = []
    original = runner.load_traceset

    def load(path):
        loads.append(path)
        return original(path)

    monkeypatch.setattr(runner, "load_traceset", load)
    assert runner.main(["--workdir", str(work), "--stages", "headroom,elicit,flips"]) == 0
    assert loads == [work / "traces/merged"]
    args = runner.parser().parse_args(flags)
    runner.cached_fingerprint(args)
    runner.cached_fingerprint(args)
    assert len(loads) == 2
    stage("collect", args, 60)
    runner.cached_fingerprint(args)
    runner.cached_fingerprint(args)
    assert len(loads) == 3


@pytest.mark.parametrize("keep", [0, 1])
def test_recovery_without_partial_and_summary_order(cloud, monkeypatch, keep):
    import shutil

    work, _, _ = cloud
    save_traceset(traces_for("p"), work / "traces/merged")

    def missing(name, args, remaining):
        shutil.rmtree(args.workdir / "traces/merged")

    monkeypatch.setattr(runner, "run_stage", missing)
    assert (
        runner.main(["--workdir", str(work), "--stages", "collect", "--keep-backups", str(keep)])
        == 0
    )
    summary = (work / "SUMMARY.md").read_text()
    assert "merged.partial-" not in summary
    assert summary.index("| collect |") < summary.index("| collect recovery |")
    if keep == 0:
        assert "merged.bak-" not in summary
