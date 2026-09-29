from __future__ import annotations

import pandas as pd

import make_fixture
from customer_retention_lab import paths
from customer_retention_lab.io import month_index


def test_generator_is_deterministic_and_fictional():
    a1, act1 = make_fixture.build(42, n_accounts=150)
    a2, act2 = make_fixture.build(42, n_accounts=150)
    assert a1.equals(a2) and act1.equals(act2)
    assert a1["account_id"].str.startswith("SYN-ACCT-").all()
    assert set(a1["plan_tier"]) <= {"tier-1", "tier-2", "tier-3"}
    a3, _ = make_fixture.build(7, n_accounts=150)
    assert not a3.equals(a1)


def test_no_activity_at_or_after_churn_month(accounts, activity):
    churned = accounts[accounts["churn_idx"].notna()]
    merged = activity.merge(churned[["account_id", "churn_idx"]], on="account_id")
    assert (merged["month_idx"] < merged["churn_idx"].astype(int)).all()
    # and activity starts at signup for every account
    first = activity.groupby("account_id")["month_idx"].min()
    assert (first.reindex(accounts["account_id"]).to_numpy() == accounts["signup_idx"].to_numpy()).all()


def test_committed_fixture_has_expected_shape(accounts, activity):
    assert len(accounts) == 3000
    assert accounts["account_id"].is_unique
    assert accounts["churn_idx"].notna().sum() > 100
    assert activity["month_idx"].max() == month_index("2025-12")
    assert activity["month_idx"].min() == month_index("2024-01")
    md = (paths.FIXTURE_DIR / "FIXTURE.md").read_text(encoding="utf-8")
    assert "Authored synthetic demonstration data" in md and "seed `42`" in md


def test_committed_accounts_match_generator():
    fresh, _ = make_fixture.build(42)
    on_disk = pd.read_csv(paths.ACCOUNTS_FIXTURE, dtype=str).fillna("")
    assert on_disk["account_id"].tolist() == fresh["account_id"].tolist()
    assert on_disk["churn_month"].tolist() == fresh["churn_month"].tolist()
