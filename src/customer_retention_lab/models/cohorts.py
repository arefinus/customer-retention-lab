"""Cohort retention curves and RFM-style summaries."""

from __future__ import annotations

import pandas as pd

from ..io import month_index, month_label


def cohort_retention(accounts: pd.DataFrame, panel_end: str) -> pd.DataFrame:
    """Long table (cohort, size, offset, active, retention). Offset k is included only
    when signup + k is inside the panel for the whole cohort, so no cohort is credited
    with retention it has not yet had the chance to lose (right-censoring)."""
    end = month_index(panel_end)
    rows = []
    for signup_idx, g in accounts.groupby("signup_idx", sort=True):
        size = int(len(g))
        max_offset = end - int(signup_idx)
        churn = g["churn_idx"]
        for k in range(0, max_offset + 1):
            month = int(signup_idx) + k
            active = int((churn.isna() | (churn > month)).sum())
            rows.append(
                {
                    "cohort": month_label(int(signup_idx)),
                    "size": size,
                    "offset": k,
                    "active": active,
                    "retention": active / size if size else None,
                }
            )
    return pd.DataFrame(rows, columns=["cohort", "size", "offset", "active", "retention"])


def cohort_table(long: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per cohort, one column per offset; blanks where censored."""
    wide = long.pivot(index="cohort", columns="offset", values="retention")
    wide.columns = [f"m{int(c)}" for c in wide.columns]
    sizes = long.groupby("cohort")["size"].first()
    return wide.assign(size=sizes).reset_index()[["cohort", "size"] + [c for c in wide.columns]]


def cohorts_for_portfolio(long: pd.DataFrame) -> list[dict]:
    out = []
    for cohort, g in long.groupby("cohort", sort=True):
        g = g.sort_values("offset")
        out.append({"cohort": cohort, "size": int(g["size"].iloc[0]), "retention": [round(float(r), 4) for r in g["retention"]]})
    return out


def rfm_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Recency (months since last activity), frequency (active months in the window) and
    monetary (invoice total in the window) quartiles by segment."""
    cols = {"recency_months": "recency", "active_months": "frequency", "invoice_total": "monetary"}
    rows = []
    for seg, g in features.groupby("segment", sort=True):
        row = {"segment": seg, "n": int(len(g)), "churn_rate": float(g["label"].mean()) if "label" in g else None}
        for col, name in cols.items():
            q = g[col].quantile([0.25, 0.5, 0.75])
            row[f"{name}_q25"], row[f"{name}_q50"], row[f"{name}_q75"] = (float(q.iloc[0]), float(q.iloc[1]), float(q.iloc[2]))
        rows.append(row)
    return pd.DataFrame(rows)
