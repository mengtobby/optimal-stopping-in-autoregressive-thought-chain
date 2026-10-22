"""Generate headroom reports from saved or synthetic traces."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from osc.analysis.headroom import headroom_report, report_to_json, report_to_markdown
from osc.data.io import load_traceset
from osc.stopping.synthetic import (
    default_model,
    exact_optimal_stop_times,
    exact_value,
    overthinking_model,
    sample_traceset,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--traces", type=Path)
    source.add_argument("--synthetic", type=int, metavar="N")
    parser.add_argument("--k-siblings", type=int, default=4)
    parser.add_argument("--model", choices=("default", "overthinking"), default="default")
    parser.add_argument("--feature-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lam", type=float, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    model = None
    if args.synthetic is not None:
        model = (
            replace(default_model(), feature_noise=args.feature_noise)
            if args.model == "default"
            else overthinking_model(args.feature_noise)
        )
    traces = (
        sample_traceset(model, args.synthetic, args.k_siblings, args.seed)
        if model is not None
        else load_traceset(args.traces)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for lam in args.lam:
        exact_times = (
            exact_optimal_stop_times(model, lam, traces)
            if model is not None and model.feature_noise == 0
            else None
        )
        report = headroom_report(traces, lam, seed=args.seed, exact_stop_times=exact_times)
        data = report_to_json(report)
        markdown = report_to_markdown(report)
        if model is not None:
            data["exact_value"] = exact_value(model, lam)
            markdown += f"\nSynthetic exact value: {data['exact_value']:.6f}.\n"
        stem = args.out / f"report_lam{lam:g}"
        stem.with_suffix(stem.suffix + ".json").write_text(
            json.dumps(data, indent=2, allow_nan=False), encoding="utf-8"
        )
        stem.with_suffix(stem.suffix + ".md").write_text(markdown, encoding="utf-8")
        logger.info(
            "lambda=%g LSM=%g dual=%g headroom=%g exact_value=%s",
            lam,
            report.lsm_lower,
            report.dual_upper,
            report.headroom,
            data.get("exact_value"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
