"""Command implementations. Return dicts; the CLI prints. Nothing here prints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .errors import MissingDataError, SchemaError
from .features import FeatureConfig, SplitConfig, build_feature_panel, eligible_observation_dates, temporal_customer_split
from .features.builder import FEATURE_COLUMNS
from .io import load_accounts, load_activity, load_table, load_yaml, month_index
from .manifest import RunManifest, make_run_id, sha256_of_file, sha256_of_mapping, utc_now, write_manifest
from .models import Calibrator, ChurnBaseline, ModelConfig, cohort_retention, cohort_table, rfm_summary, summarise
from .models.churn import MODEL_NAMES
from .models.cohorts import cohorts_for_portfolio
from .models.metrics import precision_recall_at_k
from .models.ranking import CapacityConfig, benefit_scenario, rank_at_capacity

PRIMARY_MODEL = "gbm"
SCHEMA_VERSION = 1
PRED_COLUMNS = ["account_id", "observation_date", "segment", "label"] + [f"p_{m}_{k}" for m in MODEL_NAMES for k in ("raw", "cal")]


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    return path


def load_configs(config_dir: Path = paths.CONFIG_DIR) -> dict:
    return {
        "features": load_yaml(config_dir / "features.yaml"),
        "model": load_yaml(config_dir / "model.yaml"),
        "capacity": load_yaml(config_dir / "capacity.yaml"),
    }


def run_build_features(out_dir: Path, configs: dict, accounts_path: Path = paths.ACCOUNTS_FIXTURE, activity_path: Path = paths.ACTIVITY_FIXTURE) -> dict:
    fcfg = configs["features"]
    cfg = FeatureConfig.from_mapping(fcfg)
    dates = eligible_observation_dates(cfg, str(fcfg["observation_dates"]["first"]), str(fcfg["observation_dates"]["last"]))
    accounts = load_accounts(accounts_path)
    activity = load_activity(activity_path)
    panel = build_feature_panel(accounts, activity, dates, cfg)
    split = temporal_customer_split(panel, SplitConfig.from_mapping(configs["model"]["split"], int(configs["model"]["seed"])))
    partition = pd.Series("dropped", index=panel.index)
    for name, df in split.parts().items():
        key = panel["account_id"] + "|" + panel["observation_date"]
        keys = set(df["account_id"] + "|" + df["observation_date"])
        partition[key.isin(keys)] = name
    panel = panel.assign(partition=partition)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_dir / "features_panel.csv", index=False)
    _write_json(out_dir / "split_manifest.json", {"schemaVersion": SCHEMA_VERSION, "feature_config": cfg.to_dict(), "observation_dates": dates, **split.manifest})
    return {"panel": panel, "split": split, "feature_config": cfg, "accounts": accounts, "activity": activity, "dates": dates}


def run_train(out_dir: Path, configs: dict, overrides: dict | None = None, **paths_kw) -> dict:
    built = run_build_features(out_dir, configs, **paths_kw)
    split = built["split"]
    mcfg = dict(configs["model"])
    if overrides:
        mcfg = {**mcfg, "gbm": {**mcfg.get("gbm", {}), **overrides.get("gbm", {})}}
    model_cfg = ModelConfig.from_mapping(mcfg)
    method = str(mcfg.get("calibration", {}).get("method", "isotonic"))
    X_tr, y_tr = split.train[list(FEATURE_COLUMNS)], split.train["label"].to_numpy()
    preds = {p: split.parts()[p][["account_id", "observation_date", "segment", "label"]].copy() for p in ("validation", "test")}
    exports: dict = {"schemaVersion": SCHEMA_VERSION, "models": {}, "calibration": {}}
    for name in MODEL_NAMES:
        model = ChurnBaseline(name, model_cfg).fit(X_tr, y_tr)
        p_val = model.predict_proba(split.validation[list(FEATURE_COLUMNS)])
        p_test = model.predict_proba(split.test[list(FEATURE_COLUMNS)])
        cal = Calibrator(method).fit(p_val, split.validation["label"].to_numpy())
        preds["validation"][f"p_{name}_raw"], preds["validation"][f"p_{name}_cal"] = p_val, cal.transform(p_val)
        preds["test"][f"p_{name}_raw"], preds["test"][f"p_{name}_cal"] = p_test, cal.transform(p_test)
        exports["models"][name] = model.export()
        exports["calibration"][name] = cal.to_dict()
    for p, df in preds.items():
        df[PRED_COLUMNS].to_csv(out_dir / f"predictions_{p}.csv", index=False)
    _write_json(out_dir / "models.json", exports)
    return {**built, "predictions": preds, "exports": exports, "config_hash": sha256_of_mapping(mcfg)}


def run_evaluate(predictions_path: Path, capacity: int, n_bins: int, out_dir: Path | None = None) -> dict:
    df = load_table(predictions_path, tuple(PRED_COLUMNS), "Run `python -m customer_retention_lab train` first.")
    y = df["label"].to_numpy(dtype=int)
    seg = df["segment"].astype(str).to_numpy()
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "mode": "synthetic-demo",
        "partition_file": predictions_path.name,
        "n": int(len(df)),
        "observation_dates": sorted(df["observation_date"].unique().tolist()),
        "review_capacity": capacity,
        "models": {},
    }
    bins_rows = []
    for name in MODEL_NAMES:
        result["models"][name] = {}
        for kind in ("raw", "cal"):
            label = "raw" if kind == "raw" else "calibrated"
            s = summarise(y, df[f"p_{name}_{kind}"].to_numpy(dtype=float), seg, capacity, n_bins)
            result["models"][name][label] = s
            for b in s["reliability"]:
                bins_rows.append({"model": name, "kind": label, **b})
    if out_dir is not None:
        _write_json(out_dir / "evaluation.json", result)
        pd.DataFrame(bins_rows).to_csv(out_dir / "calibration_bins.csv", index=False)
    return result


def run_rank(out_dir: Path, predictions_path: Path, capacity_cfg: CapacityConfig, model: str = PRIMARY_MODEL) -> dict:
    df = load_table(predictions_path, tuple(PRED_COLUMNS), "Run `python -m customer_retention_lab train` first.")
    scored = df.rename(columns={f"p_{model}_cal": "risk"})[["account_id", "observation_date", "segment", "risk", "label"]]
    ranked = rank_at_capacity(scored, capacity_cfg.review_capacity)
    ranked.to_csv(out_dir / "ranked_review_list.csv", index=False)
    scenario = benefit_scenario(scored, capacity_cfg)
    scenario["model"] = f"{model} (calibrated)"
    scenario["realised_at_capacity"] = _realised_at_capacity(scored, capacity_cfg.capacity_grid)
    _write_json(out_dir / "capacity_scenario.json", scenario)
    return {"ranked": ranked, "scenario": scenario, "scored": scored}


def _realised_at_capacity(scored: pd.DataFrame, grid) -> list[dict]:
    """Label-based precision and recall at k, averaged over observation dates."""
    rows = []
    for k in grid:
        per_date = []
        for _, g in scored.groupby("observation_date"):
            per_date.append(precision_recall_at_k(g["label"].to_numpy(), g["risk"].to_numpy(), int(k)))
        prec = [r["precision_at_k"] for r in per_date if r["precision_at_k"] is not None]
        rec = [r["recall_at_k"] for r in per_date if r["recall_at_k"] is not None]
        rows.append({"capacity": int(k), "precision_at_k": (float(np.mean(prec)) if prec else None), "recall_at_k": (float(np.mean(rec)) if rec else None)})
    return rows


def run_cohorts(out_dir: Path, built: dict, ranked: pd.DataFrame, scored: pd.DataFrame) -> dict:
    long = cohort_retention(built["accounts"], built["feature_config"].panel_end)
    long.to_csv(out_dir / "cohort_retention_long.csv", index=False)
    cohort_table(long).to_csv(out_dir / "cohort_table.csv", index=False)
    rfm_summary(built["panel"][built["panel"]["partition"] != "dropped"]).to_csv(out_dir / "rfm_summary.csv", index=False)
    portfolio = {
        "schemaVersion": SCHEMA_VERSION,
        "mode": "synthetic-demo",
        "generated_by": "customer_retention_lab.commands.run_cohorts",
        "cohorts": cohorts_for_portfolio(long),
        "trajectories": _trajectories(built, ranked, top_n=20),
    }
    _write_json(out_dir / "portfolio.json", portfolio)
    return {"portfolio": portfolio, "cohort_long": long}


def _trajectories(built: dict, ranked: pd.DataFrame, top_n: int) -> list[dict]:
    """Observation-window usage paths of the highest-ranked accounts at the latest test
    observation date, for the portfolio scene."""
    if ranked.empty:
        return []
    last = max(ranked["observation_date"])
    top = ranked[ranked["observation_date"] == last].nsmallest(top_n, "rank")
    T = month_index(last)
    W = built["feature_config"].observation_window
    act = built["activity"]
    out = []
    for _, r in top.iterrows():
        a = act[(act["account_id"] == r["account_id"]) & (act["month_idx"] > T - W) & (act["month_idx"] <= T)].sort_values("month_idx")
        out.append(
            {
                "account_id": r["account_id"],
                "observation_date": last,
                "rank": int(r["rank"]),
                "risk": round(float(r["risk"]), 4),
                "segment": r["segment"],
                "months": a["month"].tolist(),
                "usage": [float(v) for v in a["usage"]],
                "tickets": [int(v) for v in a["tickets"]],
            }
        )
    return out


def check_outputs(out_dir: Path) -> list[str]:
    problems: list[str] = []
    for name in ("features_panel.csv", "predictions_validation.csv", "predictions_test.csv", "models.json", "evaluation.json", "ranked_review_list.csv", "capacity_scenario.json", "cohort_table.csv", "portfolio.json"):
        if not (out_dir / name).exists():
            problems.append(f"missing {name}")
    if problems:
        return problems
    port = json.loads((out_dir / "portfolio.json").read_text(encoding="utf-8"))
    if port.get("schemaVersion") != SCHEMA_VERSION or port.get("mode") != "synthetic-demo":
        problems.append("portfolio.json: schemaVersion/mode mismatch")
    if not port.get("cohorts") or "trajectories" not in port:
        problems.append("portfolio.json: cohorts or trajectories missing")
    ev = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))
    for m in MODEL_NAMES:
        if m not in ev.get("models", {}) or "calibrated" not in ev["models"][m]:
            problems.append(f"evaluation.json: model {m} missing")
    sc = json.loads((out_dir / "capacity_scenario.json").read_text(encoding="utf-8"))
    if sc.get("causal_claim") != "none":
        problems.append("capacity_scenario.json: causal_claim must be 'none'")
    return problems


def write_run_manifest(out_dir: Path, mode: str, status: str, started: str, config_hash: str, seed: int, counts: dict, notes: tuple[str, ...] = ()) -> Path:
    data = {
        "source_id": "synthetic-fixture-v1",
        "version": "v1",
        "split_manifest_hash": sha256_of_file(out_dir / "split_manifest.json") if (out_dir / "split_manifest.json").exists() else None,
        "accounts_fixture_hash": sha256_of_file(paths.ACCOUNTS_FIXTURE) if paths.ACCOUNTS_FIXTURE.exists() else None,
        "activity_fixture_hash": sha256_of_file(paths.ACTIVITY_FIXTURE) if paths.ACTIVITY_FIXTURE.exists() else None,
        "sample_counts": counts,
    }
    m = RunManifest(
        run_id=make_run_id(mode, started), mode=mode, status=status, started_at=started, finished_at=utc_now(),
        configuration_hash=config_hash, seed=seed, data=data,
        metrics_file=_relative(out_dir / "evaluation.json"), predictions_file=_relative(out_dir / "predictions_test.csv"), notes=notes,
    )
    return write_manifest(out_dir / "run_manifest.yaml", m)


def _relative(p: Path) -> str | None:
    if not p.exists():
        return None
    try:
        return p.resolve().relative_to(paths.REPO_ROOT).as_posix()
    except ValueError:
        return p.name


def require_schema(df: pd.DataFrame, cols: tuple[str, ...], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"{what} is missing columns {missing}")
