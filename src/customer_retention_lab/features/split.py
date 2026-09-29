"""Customer-aware temporal split.

Two constraints at once:

1. accounts are partitioned into disjoint train / validation / test groups by a seeded
   hash of the account id, so no account contributes rows to two partitions;
2. observation dates are cut into three contiguous ranges, earliest for train and latest
   for test, and a row is kept only when its account's group matches its date's range.

Rows whose account group and date range disagree are dropped and counted, never moved.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from ..errors import SplitError

PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitConfig:
    account_fractions: Mapping[str, float] = field(default_factory=lambda: {"train": 0.6, "validation": 0.2, "test": 0.2})
    date_fractions: Mapping[str, float] = field(default_factory=lambda: {"train": 0.6, "validation": 0.2, "test": 0.2})
    seed: int = 42

    def __post_init__(self) -> None:
        for name, fr in (("account_fractions", self.account_fractions), ("date_fractions", self.date_fractions)):
            if set(fr) != set(PARTITIONS) or abs(sum(fr.values()) - 1.0) > 1e-9 or min(fr.values()) <= 0:
                raise SplitError(f"{name} must give positive train/validation/test shares summing to 1")

    @classmethod
    def from_mapping(cls, m: Mapping, seed: int) -> "SplitConfig":
        return cls(account_fractions=dict(m["account_fractions"]), date_fractions=dict(m["date_fractions"]), seed=seed)


@dataclass
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    manifest: dict

    def parts(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def _unit_hash(seed: int, account_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{account_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_account_partition(account_ids, cfg: SplitConfig) -> dict[str, str]:
    a, b = cfg.account_fractions["train"], cfg.account_fractions["train"] + cfg.account_fractions["validation"]
    out: dict[str, str] = {}
    for acct in account_ids:
        u = _unit_hash(cfg.seed, str(acct))
        out[str(acct)] = "train" if u < a else ("validation" if u < b else "test")
    return out


def assign_date_partition(dates: list[str], cfg: SplitConfig) -> dict[str, str]:
    ordered = sorted(set(dates))
    n = len(ordered)
    if n < 3:
        raise SplitError(f"Need at least 3 distinct observation dates for a temporal split; got {n}.")
    n_train = max(1, int(round(n * cfg.date_fractions["train"])))
    n_val = max(1, int(round(n * cfg.date_fractions["validation"])))
    if n_train + n_val >= n:
        n_train, n_val = max(1, n - 2), 1
    out: dict[str, str] = {}
    for i, d in enumerate(ordered):
        out[d] = "train" if i < n_train else ("validation" if i < n_train + n_val else "test")
    return out


def temporal_customer_split(panel: pd.DataFrame, cfg: SplitConfig) -> SplitResult:
    if "account_id" not in panel.columns or "observation_date" not in panel.columns:
        raise SplitError("panel needs account_id and observation_date columns")
    acct_part = assign_account_partition(panel["account_id"].unique(), cfg)
    date_part = assign_date_partition(panel["observation_date"].unique().tolist(), cfg)
    ap = panel["account_id"].map(acct_part)
    dp = panel["observation_date"].map(date_part)
    keep = ap == dp
    parts = {p: panel[keep & (dp == p)].reset_index(drop=True) for p in PARTITIONS}
    for p, df in parts.items():
        if df.empty:
            raise SplitError(f"Partition {p!r} received no rows; adjust fractions or observation dates.")
    manifest = {
        "seed": cfg.seed,
        "account_fractions": dict(cfg.account_fractions),
        "date_fractions": dict(cfg.date_fractions),
        "dates": {p: sorted(d for d, q in date_part.items() if q == p) for p in PARTITIONS},
        "rows": {p: int(len(df)) for p, df in parts.items()},
        "accounts": {p: int(df["account_id"].nunique()) for p, df in parts.items()},
        "positives": {p: int(df["label"].sum()) if "label" in df else None for p, df in parts.items()},
        "rows_dropped_for_disagreement": int((~keep).sum()),
    }
    return SplitResult(parts["train"], parts["validation"], parts["test"], manifest)
