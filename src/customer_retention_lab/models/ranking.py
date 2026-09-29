"""Capacity-limited review ranking and an assumed-benefit scenario.

The scenario is a planning aid, not an effect estimate: it multiplies each reviewed
account's calibrated churn probability by an assumed save rate that comes from
configuration. Nothing in this repository measures whether a review changes an outcome,
so the scenario is explicitly non-causal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ..errors import ConfigError

RANK_COLUMNS = ["observation_date", "rank", "account_id", "segment", "risk"]


@dataclass(frozen=True)
class CapacityConfig:
    review_capacity: int
    capacity_grid: tuple[int, ...]
    assumed_save_rate: float
    assumed_value_per_saved_account: float = 1.0

    def __post_init__(self) -> None:
        if self.review_capacity < 0 or any(k < 0 for k in self.capacity_grid):
            raise ConfigError("capacities must be >= 0")
        if not 0.0 <= self.assumed_save_rate <= 1.0:
            raise ConfigError("assumed_save_rate must be in [0, 1]")

    @classmethod
    def from_mapping(cls, m: Mapping) -> "CapacityConfig":
        return cls(
            review_capacity=int(m["review_capacity"]),
            capacity_grid=tuple(int(k) for k in m["capacity_grid"]),
            assumed_save_rate=float(m["assumed_save_rate"]),
            assumed_value_per_saved_account=float(m.get("assumed_value_per_saved_account", 1.0)),
        )


def rank_at_capacity(scored: pd.DataFrame, capacity: int, risk_col: str = "risk") -> pd.DataFrame:
    """Top-`capacity` accounts per observation date by descending risk. Ties are broken
    by account_id so the output is deterministic. Capacity 0 yields an empty frame."""
    if capacity < 0:
        raise ConfigError("capacity must be >= 0")
    needed = {"observation_date", "account_id", "segment", risk_col}
    missing = needed - set(scored.columns)
    if missing:
        raise ConfigError(f"scored frame is missing columns {sorted(missing)}")
    if capacity == 0 or scored.empty:
        return pd.DataFrame(columns=RANK_COLUMNS)
    df = scored.rename(columns={risk_col: "risk"}) if risk_col != "risk" else scored.copy()
    df = df.sort_values(["observation_date", "risk", "account_id"], ascending=[True, False, True])
    df["rank"] = df.groupby("observation_date").cumcount() + 1
    top = df[df["rank"] <= capacity]
    return top[RANK_COLUMNS].reset_index(drop=True)


def benefit_scenario(scored: pd.DataFrame, cfg: CapacityConfig, risk_col: str = "risk") -> dict:
    """For each capacity on the grid, the expected number of churners among the reviewed
    accounts (sum of calibrated probabilities) and the assumed saves under the
    configured save rate. Averaged per observation date."""
    dates = sorted(scored["observation_date"].unique().tolist())
    total_expected = float(scored[risk_col].sum()) / max(len(dates), 1)
    rows = []
    for k in cfg.capacity_grid:
        top = rank_at_capacity(scored, k, risk_col)
        reviewed = len(top) / max(len(dates), 1)
        expected = float(top["risk"].sum()) / max(len(dates), 1) if not top.empty else 0.0
        rows.append(
            {
                "capacity": int(k),
                "reviewed_per_date": reviewed,
                "expected_churners_reviewed": round(expected, 4),
                "share_of_expected_churn_covered": (round(expected / total_expected, 4) if total_expected > 0 else None),
                "assumed_saves": round(expected * cfg.assumed_save_rate, 4),
                "assumed_value": round(expected * cfg.assumed_save_rate * cfg.assumed_value_per_saved_account, 4),
            }
        )
    return {
        "schemaVersion": 1,
        "mode": "synthetic-demo",
        "causal_claim": "none",
        "note": (
            "Scenario simulation. assumed_saves = sum of calibrated churn probabilities among reviewed accounts "
            "x assumed_save_rate. The save rate is an input assumption, not an estimate; no outcome of any review "
            "is observed in this repository."
        ),
        "assumptions": {
            "assumed_save_rate": cfg.assumed_save_rate,
            "assumed_value_per_saved_account": cfg.assumed_value_per_saved_account,
        },
        "observation_dates": dates,
        "expected_churners_total_per_date": round(total_expected, 4),
        "grid": rows,
    }


def as_int_array(values) -> np.ndarray:
    return np.asarray(values).astype(int)
