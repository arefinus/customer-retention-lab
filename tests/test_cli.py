from __future__ import annotations

import json

import pandas as pd
import yaml

from customer_retention_lab.cli import main


def test_smoke_runs_offline_writes_manifest_and_portfolio_json(tmp_path, capsys):
    out = tmp_path / "smoke"
    code = main(["smoke", "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert "smoke ok" in printed
    manifest = yaml.safe_load((out / "run_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1 and manifest["project_id"] == "customer-retention-lab"
    assert manifest["status"] == "completed" and manifest["evidence_status"] == "demo" and manifest["git_commit"] is None
    assert manifest["metrics_file"] and "Users" not in manifest["metrics_file"]
    port = json.loads((out / "portfolio.json").read_text(encoding="utf-8"))
    assert port["schemaVersion"] == 1 and port["mode"] == "synthetic-demo"
    assert port["cohorts"] and all(c["retention"][0] == 1.0 for c in port["cohorts"])
    assert port["trajectories"] and {"account_id", "rank", "risk", "usage"} <= set(port["trajectories"][0])
    ranked = pd.read_csv(out / "ranked_review_list.csv")
    assert ranked.groupby("observation_date").size().max() <= 150
    scenario = json.loads((out / "capacity_scenario.json").read_text(encoding="utf-8"))
    assert scenario["causal_claim"] == "none"


def test_evaluate_and_rank_reuse_saved_predictions(trained, capsys):
    out = trained["out"]
    assert main(["evaluate", "--predictions", str(out / "predictions_test.csv"), "--capacity", "100"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"logistic", "gbm"} and "calibrated" in printed["gbm"]
    assert main(["rank", "--predictions", str(out / "predictions_test.csv"), "--out", str(out), "--capacity", "20"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["capacity"] == 20 and r["causal_claim"] == "none"
    ranked = pd.read_csv(out / "ranked_review_list.csv")
    assert ranked.groupby("observation_date").size().max() <= 20


def test_missing_inputs_exit_with_code_2(tmp_path, capsys):
    assert main(["evaluate", "--predictions", str(tmp_path / "nope.csv")]) == 2
    assert "not found" in capsys.readouterr().err
    assert main(["rank", "--predictions", str(tmp_path / "nope.csv"), "--out", str(tmp_path)]) == 2
    assert main(["build-features", "--accounts", str(tmp_path / "missing.csv"), "--out", str(tmp_path)]) == 2
