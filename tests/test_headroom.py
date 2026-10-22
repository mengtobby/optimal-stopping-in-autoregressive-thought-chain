"""Headroom decomposition and CLI integration checks."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from osc.analysis.headroom import (
    HeadroomError,
    headroom_report,
    report_to_json,
    report_to_markdown,
)
from osc.stopping.synthetic import default_model, exact_value, sample_traceset


def test_synthetic_bounds_and_rendering():
    model = default_model()
    report = headroom_report(sample_traceset(model, 1500, 4, 0), 0.05)
    assert report.hindsight_value >= report.dual_upper - 1e-9
    assert report.dual_upper >= report.lsm_lower - 3 * report.dual_upper_se
    assert report.dual_upper >= exact_value(model, 0.05) - 3 * report.dual_upper_se
    markdown = report_to_markdown(report)
    assert all(row.method in markdown for row in report.rows)
    data = json.loads(json.dumps(report_to_json(report), allow_nan=False))
    assert data["n_traces"] == 1500
    assert data["rows"][-1]["accuracy"] is None
    best = max(r.value for r in report.rows if r.family in ("heuristic", "learned"))
    assert report.headroom == report.dual_upper - best


def test_no_siblings_and_validation():
    traces = sample_traceset(default_model(), 30, 0, 3)
    report = headroom_report(traces, 0.05)
    assert np.isnan(report.dual_upper)
    assert any("No traces have siblings" in note for note in report.notes)
    assert all(row.family != "bound" for row in report.rows)
    assert report_to_json(report)["dual_upper"] is None
    assert "N/A" in report_to_markdown(report)
    with pytest.raises(HeadroomError, match="budget_quantiles"):
        headroom_report(traces, 0.05, budget_quantiles=())
    with pytest.raises(HeadroomError, match="convergence_ks"):
        headroom_report(traces, 0.05, convergence_ks=(0,))


def test_cli(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "headroom.py"
    spec = importlib.util.spec_from_file_location("headroom_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--synthetic", "300", "--lam", "0.05", "--out", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "report_lam0.05.json").read_text())
    assert data["exact_value"] == exact_value(default_model(), 0.05)
    assert (tmp_path / "report_lam0.05.md").is_file()
