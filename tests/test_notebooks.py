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
