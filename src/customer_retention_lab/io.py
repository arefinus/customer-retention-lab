"""Fixture loading, schema validation and month arithmetic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .errors import MissingDataError, SchemaError

ACCOUNT_COLUMNS = ("account_id", "signup_month", "plan_tier", "region", "seats", "churn_month")
ACTIVITY_COLUMNS = ("account_id", "month", "usage", "tickets", "invoice_amount", "invoice_paid")


def month_index(month: str) -> int:
    """'YYYY-MM' -> integer month count (2000-01 == 0). Raises SchemaError on bad input."""
    try:
        y, m = str(month).split("-")
        yi, mi = int(y), int(m)
    except ValueError as exc:
        raise SchemaError(f"month {month!r} is not in YYYY-MM form") from exc
    if not 1 <= mi <= 12:
        raise SchemaError(f"month {month!r} has an invalid month number")
    return (yi - 2000) * 12 + (mi - 1)


def month_label(index: int) -> str:
    y, m = divmod(int(index), 12)
    return f"{2000 + y:04d}-{m + 1:02d}"


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise MissingDataError(f"Required file not found: {path}. {hint}")


def load_yaml(path: Path) -> dict:
    _require(path, "Check the configs/ directory.")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SchemaError(f"{path} must contain a YAML mapping.")
    return data


def load_accounts(path: Path) -> pd.DataFrame:
    _require(path, "Generate it with: python tools/make_fixture.py --seed 42")
    df = pd.read_csv(path, dtype={"account_id": str, "signup_month": str, "plan_tier": str, "region": str, "churn_month": str})
    missing = [c for c in ACCOUNT_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{path} is missing columns {missing}; expected {list(ACCOUNT_COLUMNS)}.")
    if df["account_id"].duplicated().any():
        raise SchemaError(f"{path} has duplicate account_id values.")
    df["signup_idx"] = df["signup_month"].map(month_index)
    df["churn_idx"] = df["churn_month"].map(lambda v: month_index(v) if isinstance(v, str) and v else pd.NA).astype("Int64")
    return df.reset_index(drop=True)


def load_activity(path: Path) -> pd.DataFrame:
    _require(path, "Generate it with: python tools/make_fixture.py --seed 42")
    df = pd.read_csv(path, dtype={"account_id": str, "month": str})
    missing = [c for c in ACTIVITY_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{path} is missing columns {missing}; expected {list(ACTIVITY_COLUMNS)}.")
    if df.empty:
        raise SchemaError(f"{path} has no rows.")
    if (df["usage"] < 0).any() or (df["tickets"] < 0).any():
        raise SchemaError(f"{path} has negative usage or tickets.")
    df["month_idx"] = df["month"].map(month_index)
    return df.reset_index(drop=True)


def load_table(path: Path, required: tuple[str, ...], hint: str) -> pd.DataFrame:
    _require(path, hint)
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{path} is missing columns {missing}.")
    if df.empty:
        raise SchemaError(f"{path} has no rows.")
    return df
