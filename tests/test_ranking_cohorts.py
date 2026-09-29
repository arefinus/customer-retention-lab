from __future__ import annotations

import json

import pandas as pd
import pytest

from customer_retention_lab.errors import ConfigError
from customer_retention_lab.models import cohort_retention, cohort_table, rfm_summary
from customer_retention_lab.models.ranking import CapacityConfig, benefit_scenario, rank_at_capacity

CAP = CapacityConfig(review_capacity=3, capacity_grid=(1, 2, 3, 10), assumed_save_rate=0.2)


def _scored():
    return pd.DataFrame(
        {
            "observation_date": ["2025-07"] * 5 + ["2025-08"] * 4,
            "account_id": [f"SYN-ACCT-{i:05d}" for i in (5, 4, 3, 2, 1)] + [f"SYN-ACCT-{i:05d}" for i in (9, 8, 7, 6)],
            "segment": ["tier-1"] * 9,
            "risk": [0.9, 0.9, 0.5, 0.2, 0.1, 0.7, 0.6, 0.6, 0.05],
            "label": [1, 0, 1, 0, 0, 1, 0, 0, 0],
        }
    )


def test_capacity_is_respected_per_observation_date():
    ranked = rank_at_capacity(_scored(), 3)
    counts = ranked.groupby("observation_date").size()
    assert counts["2025-07"] == 3 and counts["2025-08"] == 3
    assert (ranked["rank"] <= 3).all()
    assert rank_at_capacity(_scored(), 100).groupby("observation_date").size().tolist() == [5, 4]


def test_ranking_is_descending_with_deterministic_tie_break():
    ranked = rank_at_capacity(_scored(), 5)
    jul = ranked[ranked["observation_date"] == "2025-07"]
    assert jul["risk"].is_monotonic_decreasing
    assert jul["account_id"].tolist()[:2] == ["SYN-ACCT-00004", "SYN-ACCT-00005"]  # tie at 0.9 -> smaller id first
    assert rank_at_capacity(_scored(), 5).equals(rank_at_capacity(_scored(), 5))


def test_zero_capacity_and_validation():
    assert rank_at_capacity(_scored(), 0).empty
    with pytest.raises(ConfigError):
        rank_at_capacity(_scored(), -1)
    with pytest.raises(ConfigError):
        rank_at_capacity(_scored().drop(columns=["segment"]), 2)
    with pytest.raises(ConfigError):
        CapacityConfig(review_capacity=1, capacity_grid=(1,), assumed_save_rate=1.5)


def test_benefit_scenario_is_labelled_non_causal_and_monotone():
    sc = benefit_scenario(_scored(), CAP)
    assert sc["causal_claim"] == "none" and "assumed_save_rate" in sc["assumptions"]
    saves = [r["assumed_saves"] for r in sc["grid"]]
    assert saves == sorted(saves)
    full = sc["grid"][-1]
    assert full["share_of_expected_churn_covered"] == pytest.approx(1.0)
    assert full["assumed_saves"] == pytest.approx(full["expected_churners_reviewed"] * 0.2)
    json.dumps(sc)


def test_cohort_retention_starts_at_one_and_never_increases(accounts):
    long = cohort_retention(accounts, "2025-12")
    for cohort, g in long.groupby("cohort"):
        g = g.sort_values("offset")
        assert g["retention"].iloc[0] == 1.0
        assert g["retention"].is_monotonic_decreasing
    assert long.groupby("cohort")["size"].first().sum() == len(accounts)


def test_cohort_offsets_are_right_censored_at_panel_end(accounts):
    long = cohort_retention(accounts, "2025-12")
    max_off = long.groupby("cohort")["offset"].max()
    assert max_off["2025-06"] == 6 and max_off["2024-01"] == 23
    wide = cohort_table(long)
    assert wide.loc[wide["cohort"] == "2025-06", "m23"].isna().all()
    assert wide.loc[wide["cohort"] == "2024-01", "m23"].notna().all()


def test_rfm_summary_by_segment(trained):
    panel = trained["panel"]
    summary = rfm_summary(panel[panel["partition"] != "dropped"])
    assert set(summary["segment"]) == set(panel["segment"].unique())
    assert (summary["recency_q25"] <= summary["recency_q75"]).all()
    assert summary["n"].sum() == (panel["partition"] != "dropped").sum()
