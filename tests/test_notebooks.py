import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("platform", "limit", "hours", "workdirs"),
    [
        ("kaggle", "60", "11", ["/kaggle/working/osc-runs/math500-r2"]),
        (
            "colab",
            "20",
            "3",
            ["/content/drive/MyDrive/osc-runs/math500-r2", "/content/osc-runs/math500-r2"],
        ),
    ],
)
def test_cloud_run_settings(platform: str, limit: str, hours: str, workdirs: list[str]) -> None:
    notebook = json.loads((ROOT / "notebooks" / f"{platform}_run.ipynb").read_text())
    assert notebook["nbformat"] == 4
    assert [cell["id"] for cell in notebook["cells"]] == [
        f"pipeline-cell-{index}" for index in range(6)
    ]
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    run = ast.parse(cells["pipeline-cell-4"])
    command = run.body[0].value
    assert isinstance(command, ast.List)
    assert ast.literal_eval(command.elts[1]) == "scripts/run_pipeline.py"
    assert ast.literal_eval(command.elts[2]) == "--workdir"
    assert ast.unparse(command.elts[3]) == "str(workdir)"
    arguments = [ast.literal_eval(item) for item in command.elts[4:]]
    flags = dict(zip(arguments[::2], arguments[1::2], strict=True))
    assert flags == {
        "--model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "--dataset": "math500",
        "--limit": limit,
        "--max-hours": hours,
        "--dtype": "float16",
        "--n-forced": "2",
        "--k-siblings": "4",
        "--sibling-stride": "2",
        "--sibling-max-tokens": "128",
        "--max-thinking-tokens": "3072",
        "--max-steps": "20",
    }
    setup = ast.parse(cells["pipeline-cell-3"])
    paths = [
        ast.literal_eval(node.args[0])
        for node in ast.walk(setup)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path"
    ]
    assert sorted(paths) == sorted(workdirs)
    docs = (ROOT / "docs" / "cloud.md").read_text()
    for workdir in workdirs:
        assert f"`{workdir}`" in docs


def test_kaggle_pilot_settings() -> None:
    notebook = json.loads((ROOT / "notebooks" / "kaggle_pilots.ipynb").read_text())
    original = json.loads((ROOT / "notebooks" / "kaggle_run.ipynb").read_text())
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids)) == 6
    assert set(ids).isdisjoint(cell["id"] for cell in original["cells"])
    assert notebook["cells"][2]["source"] == original["cells"][2]["source"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    run = ast.parse("".join(notebook["cells"][4]["source"]))
    command = run.body[0].value
    assert isinstance(command, ast.List)
    assert ast.literal_eval(command.elts[1]) == "scripts/run_pipeline.py"
    assert ast.literal_eval(command.elts[2]) == "--workdir"
    assert ast.unparse(command.elts[3]) == "str(workdir)"
    arguments = [ast.literal_eval(item) for item in command.elts[4:]]
    assert dict(zip(arguments[::2], arguments[1::2], strict=True)) == {
        "--model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "--dataset": "math500",
        "--limit": "300",
        "--max-hours": "11",
        "--dtype": "float16",
        "--n-forced": "4",
        "--k-siblings": "0",
        "--max-thinking-tokens": "3072",
        "--max-steps": "20",
    }
    setup = ast.parse("".join(notebook["cells"][3]["source"]))
    assert ast.literal_eval(setup.body[0].value.args[0]) == (
        "/kaggle/working/osc-runs/math500-pilots"
    )
    docs = (ROOT / "docs" / "cloud.md").read_text()
    assert "`/kaggle/working/osc-runs/math500-pilots`" in docs


def test_kaggle_idea3_settings() -> None:
    notebook = json.loads((ROOT / "notebooks" / "kaggle_idea3.ipynb").read_text())
    original = json.loads((ROOT / "notebooks" / "kaggle_pilots.ipynb").read_text())
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids)) == 6
    assert ids == [cell["id"] for cell in original["cells"]]
    assert notebook["cells"][2]["source"] == original["cells"][2]["source"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    run = ast.parse("".join(notebook["cells"][4]["source"]))
    command = run.body[0].value
    assert isinstance(command, ast.List)
    assert ast.literal_eval(command.elts[1]) == "scripts/run_pipeline.py"
    assert ast.literal_eval(command.elts[2]) == "--workdir"
    assert ast.unparse(command.elts[3]) == "str(workdir)"
    arguments = [ast.literal_eval(item) for item in command.elts[4:]]
    assert dict(zip(arguments[::2], arguments[1::2], strict=True)) == {
        "--model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "--dataset": "math500",
        "--limit": "300",
        "--max-hours": "11",
        "--dtype": "float16",
        "--n-forced": "1",
        "--k-siblings": "0",
        "--max-thinking-tokens": "4096",
        "--max-steps": "64",
        "--elicit-max-k": "64",
    }
    setup = ast.parse("".join(notebook["cells"][3]["source"]))
    assert ast.literal_eval(setup.body[0].value.args[0]) == (
        "/kaggle/working/osc-runs/math500-idea3"
    )
    docs = (ROOT / "docs" / "cloud.md").read_text()
    assert "`/kaggle/working/osc-runs/math500-idea3`" in docs
