"""Point-in-time features.

For an observation date T (a month), features are computed from activity in the
observation window [T - W + 1, T] and nothing later. The outcome is whether the
account's churn month falls in the outcome window (T, T + K]. The two windows are
disjoint by construction: the activity frame is filtered to month_idx <= T before any
aggregation, so a feature that reads the outcome window cannot be written here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ..errors import CensoringError, ConfigError
from ..io import month_index, month_label

NUMERIC_FEATURES = (
    "tenure_months",
    "seats",
    "usage_mean",
    "usage_last",
    "usage_trend",
    "tickets_sum",
    "tickets_last",
    "active_months",
    "recency_months",
    "invoice_total",
    "paid_ratio",
)
CATEGORICAL_FEATURES = ("plan_tier", "region")
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
META_COLUMNS = ("account_id", "observation_date", "observation_idx", "label", "segment")


@dataclass(frozen=True)
class FeatureConfig:
    panel_start: str
    panel_end: str
    observation_window: int = 6
    outcome_window: int = 3
    min_tenure: int = 1

    def __post_init__(self) -> None:
        if self.observation_window < 1 or self.outcome_window < 1 or self.min_tenure < 1:
            raise ConfigError("observation_window, outcome_window and min_tenure must be >= 1")
        if month_index(self.panel_end) < month_index(self.panel_start):
            raise ConfigError("panel_end precedes panel_start")

    @classmethod
    def from_mapping(cls, m: Mapping) -> "FeatureConfig":
        return cls(
            panel_start=str(m["panel_start"]),
            panel_end=str(m["panel_end"]),
            observation_window=int(m.get("observation_window_months", 6)),
            outcome_window=int(m.get("outcome_window_months", 3)),
            min_tenure=int(m.get("min_tenure_months", 1)),
        )

    def to_dict(self) -> dict:
        return {
            "panel_start": self.panel_start,
            "panel_end": self.panel_end,
            "observation_window_months": self.observation_window,
            "outcome_window_months": self.outcome_window,
            "min_tenure_months": self.min_tenure,
        }


def eligible_observation_dates(cfg: FeatureConfig, first: str, last: str) -> list[str]:
    """Monthly observation dates from first to last whose outcome window ends inside the panel."""
    lo, hi = month_index(first), month_index(last)
    end = month_index(cfg.panel_end)
    dates = [month_label(t) for t in range(lo, hi + 1) if t + cfg.outcome_window <= end]
    if not dates:
        raise CensoringError(
            f"No observation date in {first}..{last} has a full {cfg.outcome_window}-month outcome window before {cfg.panel_end}."
        )
    return dates


def build_features(accounts: pd.DataFrame, activity: pd.DataFrame, observation_date: str, cfg: FeatureConfig) -> pd.DataFrame:
    T = month_index(observation_date)
    end = month_index(cfg.panel_end)
    if T + cfg.outcome_window > end:
        raise CensoringError(
            f"Observation date {observation_date} needs the outcome window through {month_label(T + cfg.outcome_window)}, "
            f"but the panel ends {cfg.panel_end}; accounts observed at this date are censored and must be excluded."
        )
    if T < month_index(cfg.panel_start):
        raise CensoringError(f"Observation date {observation_date} precedes the panel start {cfg.panel_start}.")

    churn_idx = accounts["churn_idx"]
    at_risk = (accounts["signup_idx"] <= T - (cfg.min_tenure - 1)) & (churn_idx.isna() | (churn_idx > T))
    base = accounts.loc[at_risk, ["account_id", "signup_idx", "churn_idx", "plan_tier", "region", "seats"]].copy()
    if base.empty:
        return pd.DataFrame(columns=list(META_COLUMNS) + list(FEATURE_COLUMNS))

    # The only activity that may influence a feature: observation window, closed at T.
    window = activity[(activity["month_idx"] >= T - cfg.observation_window + 1) & (activity["month_idx"] <= T)]
    window = window[window["account_id"].isin(base["account_id"])]

    agg = _aggregate_window(window, T)
    out = base.merge(agg, on="account_id", how="left")
    out = _fill_missing(out, cfg)
    out["tenure_months"] = (T - out["signup_idx"] + 1).astype(int)
    out["label"] = ((out["churn_idx"] > T) & (out["churn_idx"] <= T + cfg.outcome_window)).fillna(False).astype(int)
    out["observation_date"] = observation_date
    out["observation_idx"] = T
    out["segment"] = out["plan_tier"]
    return out[list(META_COLUMNS) + list(FEATURE_COLUMNS)].reset_index(drop=True)


def _aggregate_window(window: pd.DataFrame, T: int) -> pd.DataFrame:
    if window.empty:
        return pd.DataFrame(columns=["account_id"])
    w = window.sort_values(["account_id", "month_idx"])
    g = w.groupby("account_id", sort=False)
    last_idx = g["month_idx"].max()
    usage_last = g["usage"].last()
    # A month with no row counts as zero usage, so both means divide by three months.
    accounts = pd.Index(g.size().index)
    recent = (w[w["month_idx"] > T - 3].groupby("account_id")["usage"].sum().reindex(accounts).fillna(0.0)) / 3.0
    prior = (w[w["month_idx"] <= T - 3].groupby("account_id")["usage"].sum().reindex(accounts).fillna(0.0)) / 3.0
    trend = recent - prior
    agg = pd.DataFrame(
        {
            "usage_mean": g["usage"].mean(),
            "usage_last": usage_last,
            "usage_trend": trend.reindex(usage_last.index).fillna(0.0),
            "tickets_sum": g["tickets"].sum(),
            "tickets_last": g["tickets"].last(),
            "active_months": g["month_idx"].count(),
            "recency_months": (T - last_idx),
            "invoice_total": g["invoice_amount"].sum(),
            "paid_ratio": g["invoice_paid"].mean(),
        }
    )
    return agg.reset_index()


def _fill_missing(out: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Accounts at risk but with no activity in the window: zero activity, maximal recency."""
    defaults = {
        "usage_mean": 0.0,
        "usage_last": 0.0,
        "usage_trend": 0.0,
        "tickets_sum": 0,
        "tickets_last": 0,
        "active_months": 0,
        "recency_months": cfg.observation_window,
        "invoice_total": 0.0,
        "paid_ratio": 1.0,
    }
    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val
        out[col] = out[col].fillna(val)
    out["tickets_sum"] = out["tickets_sum"].astype(int)
    out["tickets_last"] = out["tickets_last"].astype(int)
    out["active_months"] = out["active_months"].astype(int)
    out["recency_months"] = out["recency_months"].astype(int)
    return out


def build_feature_panel(accounts: pd.DataFrame, activity: pd.DataFrame, dates: list[str], cfg: FeatureConfig) -> pd.DataFrame:
    frames = [build_features(accounts, activity, d, cfg) for d in dates]
    panel = pd.concat(frames, ignore_index=True)
    if panel.empty:
        raise CensoringError("Feature panel is empty: no account is at risk on any requested observation date.")
    return panel
