"""argparse entry point: python -m customer_retention_lab <subcommand>.

Exit codes: 0 ok; 1 failure or schema mismatch; 2 missing input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import paths
from .errors import MissingDataError, RetentionLabError
from .manifest import utc_now

SMOKE_OVERRIDES = {"gbm": {"max_iter": 40}}


def _p(msg: str) -> None:
    sys.stdout.write(msg + "\n")


def _pipeline(out: Path, overrides: dict | None, with_figures: bool, figures_dir: Path | None, mode: str) -> int:
    from . import commands, figures
    from .models.ranking import CapacityConfig

    started = utc_now()
    configs = commands.load_configs()
    trained = commands.run_train(out, configs, overrides=overrides)
    cap = CapacityConfig.from_mapping(configs["capacity"])
    n_bins = int(configs["model"].get("reliability_bins", 10))
    evaluation = commands.run_evaluate(out / "predictions_test.csv", cap.review_capacity, n_bins, out_dir=out)
    ranked = commands.run_rank(out, out / "predictions_test.csv", cap)
    commands.run_cohorts(out, trained, ranked["ranked"], ranked["scored"])
    problems = commands.check_outputs(out)
    counts = {**trained["split"].manifest["rows"], "accounts": int(len(trained["accounts"])), "activity_rows": int(len(trained["activity"]))}
    status = "completed" if not problems else "failed"
    commands.write_run_manifest(out, mode, status, started, trained["config_hash"], int(configs["model"]["seed"]), counts,
                                notes=(("smoke uses fewer boosting iterations than demo",) if overrides else ()))
    for pr in problems:
        _p(f"SCHEMA PROBLEM: {pr}")
    if problems:
        return 1
    if with_figures:
        figs = figures.render_all(out, figures_dir or paths.DOCS_FIGURE_DIR, commands.PRIMARY_MODEL)
        _p(f"figures: {len(figs)} -> {figures_dir or paths.DOCS_FIGURE_DIR}")
    m = trained["split"].manifest
    _p(f"{mode} ok: rows train/val/test = {m['rows']['train']}/{m['rows']['validation']}/{m['rows']['test']}, "
       f"positives = {m['positives']['train']}/{m['positives']['validation']}/{m['positives']['test']}, dropped = {m['rows_dropped_for_disagreement']}")
    for name, v in evaluation["models"].items():
        for kind in ("raw", "calibrated"):
            s = v[kind]
            _p(f"{name:9s} {kind:10s} brier={_fmt(s['brier'])} auroc={_fmt(s['auroc'])} ap={_fmt(s['average_precision'])} "
               f"precision@{cap.review_capacity}={_fmt(s['at_capacity']['precision_at_k'])} recall@{cap.review_capacity}={_fmt(s['at_capacity']['recall_at_k'])}")
    _p(f"ranked review list: {len(ranked['ranked'])} rows -> {out / 'ranked_review_list.csv'}")
    _p("All numbers above were computed on synthetic fixture v1 (seed 42); they are not benchmark results.")
    return 0


def _fmt(v) -> str:
    return "undefined" if v is None else f"{v:.3f}"


def cmd_smoke(args: argparse.Namespace) -> int:
    return _pipeline(Path(args.out), SMOKE_OVERRIDES, with_figures=False, figures_dir=None, mode="smoke")


def cmd_demo(args: argparse.Namespace) -> int:
    return _pipeline(Path(args.out), None, with_figures=True, figures_dir=Path(args.figures), mode="demo")


def cmd_build_features(args: argparse.Namespace) -> int:
    from . import commands

    built = commands.run_build_features(Path(args.out), commands.load_configs(), accounts_path=Path(args.accounts), activity_path=Path(args.activity))
    _p(json.dumps(built["split"].manifest, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from . import commands

    trained = commands.run_train(Path(args.out), commands.load_configs(), accounts_path=Path(args.accounts), activity_path=Path(args.activity))
    _p(json.dumps({"rows": trained["split"].manifest["rows"], "calibration": {k: {"method": v["method"], "n_fit": v["n_fit"]} for k, v in trained["exports"]["calibration"].items()}}, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from . import commands

    configs = commands.load_configs()
    res = commands.run_evaluate(Path(args.predictions), int(args.capacity or configs["capacity"]["review_capacity"]), int(configs["model"].get("reliability_bins", 10)),
                                out_dir=(Path(args.out) if args.out else None))
    _p(json.dumps({m: {k: {kk: v[k][kk] for kk in ("n", "brier", "auroc", "average_precision")} for k in v} for m, v in res["models"].items()}, indent=2))
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    from . import commands
    from .models.ranking import CapacityConfig

    cap = CapacityConfig.from_mapping(commands.load_configs()["capacity"])
    if args.capacity is not None:
        cap = CapacityConfig(int(args.capacity), cap.capacity_grid, cap.assumed_save_rate, cap.assumed_value_per_saved_account)
    res = commands.run_rank(Path(args.out), Path(args.predictions), cap)
    _p(json.dumps({"rows": len(res["ranked"]), "capacity": cap.review_capacity, "causal_claim": res["scenario"]["causal_claim"]}, indent=2))
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from . import commands, figures

    for p in figures.render_all(Path(args.out), Path(args.figures), commands.PRIMARY_MODEL):
        _p(f"wrote {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m customer_retention_lab", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    def add_data_args(p):
        p.add_argument("--accounts", default=str(paths.ACCOUNTS_FIXTURE))
        p.add_argument("--activity", default=str(paths.ACTIVITY_FIXTURE))

    s = sub.add_parser("smoke", help="fast offline pipeline on the fixture; exit 1 on schema mismatch")
    s.add_argument("--out", default=str(paths.OUTPUT_DIR / "smoke"))
    s.set_defaults(func=cmd_smoke)

    d = sub.add_parser("demo", help="full pipeline; writes examples/output/ and docs/figures/")
    d.add_argument("--out", default=str(paths.OUTPUT_DIR))
    d.add_argument("--figures", default=str(paths.DOCS_FIGURE_DIR))
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser("build-features", help="point-in-time features and the customer-aware temporal split")
    b.add_argument("--out", default=str(paths.OUTPUT_DIR))
    add_data_args(b)
    b.set_defaults(func=cmd_build_features)

    t = sub.add_parser("train", help="fit both baselines on train, calibrate on validation, write predictions")
    t.add_argument("--out", default=str(paths.OUTPUT_DIR))
    add_data_args(t)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a saved predictions file")
    e.add_argument("--predictions", default=str(paths.OUTPUT_DIR / "predictions_test.csv"))
    e.add_argument("--capacity", type=int, default=None)
    e.add_argument("--out", default=None, help="directory for evaluation.json and calibration_bins.csv")
    e.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("rank", help="ranked review list at capacity plus the assumed-benefit scenario")
    r.add_argument("--predictions", default=str(paths.OUTPUT_DIR / "predictions_test.csv"))
    r.add_argument("--out", default=str(paths.OUTPUT_DIR))
    r.add_argument("--capacity", type=int, default=None)
    r.set_defaults(func=cmd_rank)

    f = sub.add_parser("figures", help="render docs/figures/*.svg from examples/output/")
    f.add_argument("--out", default=str(paths.OUTPUT_DIR))
    f.add_argument("--figures", default=str(paths.DOCS_FIGURE_DIR))
    f.set_defaults(func=cmd_figures)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except MissingDataError as exc:
        sys.stderr.write(f"missing input: {exc}\n")
        return 2
    except RetentionLabError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
