from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import tiny_panel
from customer_retention_lab.errors import CensoringError, ConfigError
from customer_retention_lab.features import FeatureConfig, build_feature_panel, build_features, eligible_observation_dates
from customer_retention_lab.features.builder import FEATURE_COLUMNS
from customer_retention_lab.io import month_index

T = "2024-06"


def test_features_ignore_outcome_window(accounts, activity, cfg):
    base = build_features(accounts, activity, "2025-03", cfg)
    corrupted = activity.copy()
    future = corrupted["month_idx"] > month_index("2025-03")
    corrupted.loc[future, ["usage", "tickets", "invoice_amount"]] = 1e9
    corrupted.loc[future, "invoice_paid"] = 0
    again = build_features(accounts, corrupted, "2025-03", cfg)
    pd.testing.assert_frame_equal(base, again)


def test_features_ignore_months_before_the_observation_window(accounts, activity, cfg):
    base = build_features(accounts, activity, "2025-03", cfg)
    corrupted = activity.copy()
    early = corrupted["month_idx"] < month_index("2025-03") - cfg.observation_window + 1
    corrupted.loc[early, ["usage", "tickets"]] = 1e9
    pd.testing.assert_frame_equal(base, build_features(accounts, corrupted, "2025-03", cfg))


def test_label_is_churn_inside_the_future_window_only(cfg):
    accounts, activity = tiny_panel()
    f = build_features(accounts, activity, T, cfg).set_index("account_id")
    assert f.loc["SYN-ACCT-A", "label"] == 1      # churn 2024-08 in (06, 09]
    assert f.loc["SYN-ACCT-B", "label"] == 0      # churn 2024-10 after the window
    assert f.loc["SYN-ACCT-D", "label"] == 0      # never churns
    assert "SYN-ACCT-C" not in f.index            # churned before T: not at risk
    assert "SYN-ACCT-E" not in f.index            # signed up after T


def test_censored_observation_dates_are_refused(cfg):
    accounts, activity = tiny_panel()
    with pytest.raises(CensoringError):
        build_features(accounts, activity, "2025-10", cfg)     # 2025-10 + 3 > 2025-12
    with pytest.raises(CensoringError):
        build_features(accounts, activity, "2023-12", cfg)
    dates = eligible_observation_dates(cfg, "2025-06", "2025-12")
    assert dates == ["2025-06", "2025-07", "2025-08", "2025-09"]
    with pytest.raises(CensoringError):
        eligible_observation_dates(cfg, "2025-10", "2025-12")


def test_exact_feature_values_on_tiny_panel(cfg):
    accounts, activity = tiny_panel()
    f = build_features(accounts, activity, T, cfg).set_index("account_id")
    a = f.loc["SYN-ACCT-A"]
    assert a["tenure_months"] == 6 and a["active_months"] == 6 and a["recency_months"] == 0
    assert a["usage_mean"] == 10.0 and a["tickets_sum"] == 6 and a["invoice_total"] == 120.0 and a["paid_ratio"] == 1.0
    b = f.loc["SYN-ACCT-B"]
    assert b["tenure_months"] == 4 and b["active_months"] == 2 and b["recency_months"] == 2
    # B has one row at 10 in the prior three months (Mar) and one at 10 in the recent three (Apr);
    # months without a row count as zero, so both three-month means are 10/3 and the trend is 0.
    assert b["usage_trend"] == pytest.approx(0.0)
    assert list(f.columns) == ["observation_date", "observation_idx", "label", "segment"] + list(FEATURE_COLUMNS)


def test_accounts_with_no_window_activity_get_defaults(cfg):
    accounts, activity = tiny_panel()
    activity = activity[activity["account_id"] != "SYN-ACCT-D"]  # D at risk but silent
    f = build_features(accounts, activity, T, cfg).set_index("account_id")
    d = f.loc["SYN-ACCT-D"]
    assert d["active_months"] == 0 and d["recency_months"] == cfg.observation_window and d["usage_mean"] == 0.0 and d["paid_ratio"] == 1.0


def test_min_tenure_excludes_brand_new_accounts():
    accounts, activity = tiny_panel()
    cfg2 = FeatureConfig(panel_start="2024-01", panel_end="2025-12", min_tenure=6)
    f = build_features(accounts, activity, T, cfg2)
    assert set(f["account_id"]) == {"SYN-ACCT-A", "SYN-ACCT-D"}  # B has tenure 4 at T


def test_panel_over_dates_and_config_validation(accounts, activity, cfg):
    panel = build_feature_panel(accounts, activity, ["2025-01", "2025-02"], cfg)
    assert set(panel["observation_date"]) == {"2025-01", "2025-02"}
    assert panel[list(FEATURE_COLUMNS)].isna().sum().sum() == 0
    assert panel["label"].isin([0, 1]).all()
    with pytest.raises(ConfigError):
        FeatureConfig(panel_start="2024-01", panel_end="2025-12", observation_window=0)
    with pytest.raises(ConfigError):
        FeatureConfig(panel_start="2025-01", panel_end="2024-12")


def test_no_feature_correlates_perfectly_with_label_on_fixture(accounts, activity, cfg):
    """A feature that encodes the outcome would show near-perfect separation."""
    panel = build_features(accounts, activity, "2025-03", cfg)
    num = panel[[c for c in FEATURE_COLUMNS if c not in ("plan_tier", "region")]].astype(float)
    corr = num.corrwith(panel["label"].astype(float)).abs().fillna(0.0)
    assert corr.max() < 0.9, corr.sort_values(ascending=False).head(3).to_dict()
    assert np.isfinite(num.to_numpy()).all()
