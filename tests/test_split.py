from __future__ import annotations

import pandas as pd
import pytest

from customer_retention_lab.errors import SplitError
from customer_retention_lab.features import SplitConfig, assign_account_partition, build_feature_panel, temporal_customer_split


@pytest.fixture(scope="module")
def panel(accounts, activity):
    from conftest import CFG

    return build_feature_panel(accounts, activity, [f"2025-0{m}" for m in range(1, 10)], CFG)


def test_no_account_appears_in_two_partitions(panel):
    res = temporal_customer_split(panel, SplitConfig())
    ids = {p: set(df["account_id"]) for p, df in res.parts().items()}
    assert not (ids["train"] & ids["validation"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["validation"] & ids["test"])


def test_observation_dates_are_ordered_train_before_validation_before_test(panel):
    res = temporal_customer_split(panel, SplitConfig())
    assert max(res.train["observation_date"]) < min(res.validation["observation_date"])
    assert max(res.validation["observation_date"]) < min(res.test["observation_date"])
    assert res.manifest["dates"]["train"] == sorted(res.train["observation_date"].unique())


def test_rows_are_kept_or_dropped_never_moved(panel):
    res = temporal_customer_split(panel, SplitConfig())
    kept = sum(res.manifest["rows"].values())
    assert kept + res.manifest["rows_dropped_for_disagreement"] == len(panel)
    assert res.manifest["rows_dropped_for_disagreement"] > 0
    for name, df in res.parts().items():
        merged = df.merge(panel, on=["account_id", "observation_date"], how="left", suffixes=("", "_orig"))
        assert (merged["label"] == merged["label_orig"]).all()


def test_split_is_deterministic_and_seed_sensitive(panel):
    a = temporal_customer_split(panel, SplitConfig(seed=42))
    b = temporal_customer_split(panel, SplitConfig(seed=42))
    c = temporal_customer_split(panel, SplitConfig(seed=43))
    pd.testing.assert_frame_equal(a.test, b.test)
    assert set(a.test["account_id"]) != set(c.test["account_id"])
    part = assign_account_partition(["SYN-ACCT-00001", "SYN-ACCT-00002"], SplitConfig())
    assert set(part.values()) <= {"train", "validation", "test"}


def test_account_shares_roughly_follow_fractions(panel):
    part = assign_account_partition(panel["account_id"].unique(), SplitConfig())
    s = pd.Series(part).value_counts(normalize=True)
    assert abs(s["train"] - 0.6) < 0.05 and abs(s["validation"] - 0.2) < 0.05 and abs(s["test"] - 0.2) < 0.05


def test_split_rejects_too_few_dates_and_bad_fractions(panel):
    two = panel[panel["observation_date"].isin(["2025-01", "2025-02"])]
    with pytest.raises(SplitError):
        temporal_customer_split(two, SplitConfig())
    with pytest.raises(SplitError):
        SplitConfig(account_fractions={"train": 0.5, "validation": 0.5, "test": 0.5})
    with pytest.raises(SplitError):
        temporal_customer_split(panel.drop(columns=["observation_date"]), SplitConfig())
