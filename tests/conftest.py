from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from customer_retention_lab import commands, paths  # noqa: E402
from customer_retention_lab.features import FeatureConfig  # noqa: E402
from customer_retention_lab.io import load_accounts, load_activity  # noqa: E402

CFG = FeatureConfig(panel_start="2024-01", panel_end="2025-12", observation_window=6, outcome_window=3)


@pytest.fixture(scope="session")
def accounts() -> pd.DataFrame:
    return load_accounts(paths.ACCOUNTS_FIXTURE)


@pytest.fixture(scope="session")
def activity() -> pd.DataFrame:
    return load_activity(paths.ACTIVITY_FIXTURE)


@pytest.fixture
def cfg() -> FeatureConfig:
    return CFG


@pytest.fixture(scope="session")
def trained(tmp_path_factory) -> dict:
    """One fast end-to-end training run shared by model, ranking and CLI tests."""
    out = tmp_path_factory.mktemp("trained")
    configs = commands.load_configs()
    return {"out": out, "configs": configs, **commands.run_train(out, configs, overrides={"gbm": {"max_iter": 40}})}


def tiny_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hand-built accounts and activity for exact-value feature tests."""
    accounts = pd.DataFrame(
        {
            "account_id": ["SYN-ACCT-A", "SYN-ACCT-B", "SYN-ACCT-C", "SYN-ACCT-D", "SYN-ACCT-E"],
            "signup_month": ["2024-01", "2024-03", "2024-01", "2024-01", "2024-09"],
            "plan_tier": ["tier-1", "tier-2", "tier-1", "tier-3", "tier-1"],
            "region": ["region-A", "region-B", "region-A", "region-C", "region-D"],
            "seats": [3, 5, 2, 9, 1],
            # A churns inside the outcome window of T=2024-06 (2024-08); B churns after it (2024-10);
            # C churned before T (2024-05); D never churns; E signs up after T.
            "churn_month": ["2024-08", "2024-10", "2024-05", "", ""],
        }
    )
    rows = []
    for acct, start, end in (("SYN-ACCT-A", "2024-01", "2024-07"), ("SYN-ACCT-B", "2024-03", "2024-09"), ("SYN-ACCT-C", "2024-01", "2024-04"), ("SYN-ACCT-D", "2024-01", "2025-12"), ("SYN-ACCT-E", "2024-09", "2025-12")):
        for m in pd.period_range(start, end, freq="M"):
            rows.append({"account_id": acct, "month": str(m), "usage": 10.0, "tickets": 1, "invoice_amount": 20.0, "invoice_paid": 1})
    activity = pd.DataFrame(rows)
    # B has no activity in the last two window months before T=2024-06 (May, June removed) -> recency 2
    activity = activity[~((activity["account_id"] == "SYN-ACCT-B") & (activity["month"].isin(["2024-05", "2024-06"])))]
    return _prepare(accounts, activity)


def _prepare(accounts: pd.DataFrame, activity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from customer_retention_lab.io import month_index

    accounts = accounts.copy()
    accounts["signup_idx"] = accounts["signup_month"].map(month_index)
    accounts["churn_idx"] = accounts["churn_month"].map(lambda v: month_index(v) if v else pd.NA).astype("Int64")
    activity = activity.copy()
    activity["month_idx"] = activity["month"].map(month_index)
    return accounts, activity.reset_index(drop=True)
